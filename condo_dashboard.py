from pathlib import Path
import io
import hashlib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupShuffleSplit
from scipy.optimize import lsq_linear

st.set_page_config(page_title='Condo Valuation Studio',page_icon=None,layout='wide')
st.markdown("""<style>
/* Condo Valuation Studio | Institutional visual system */
:root { --ink:#15253b; --muted:#61748b; --navy:#153653; --blue:#24628b; --line:#dce5ed; --surface:#f5f8fb; }
html, body, [class*="css"] {font-family: 'Segoe UI', Inter, -apple-system, BlinkMacSystemFont, sans-serif;}
[data-testid="stAppViewContainer"] {background: #f7f9fc;}
[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1480px; padding: 2.0rem 2.7rem 3.5rem;}
[data-testid="stSidebar"] {background: #eef3f8; border-right: 1px solid #d9e3ec;}
[data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
h1,h2,h3 {color:var(--ink); letter-spacing:-.035em; font-weight:650;}
h1 {font-size: 2.05rem !important; margin-bottom:.15rem !important;}
h2,h3 {padding-top:.5rem;}
p, label, .stCaption {color:#33485f;}
[data-testid="stCaptionContainer"] p {color:var(--muted);font-size:.84rem;}
[data-testid="stMetric"] {background:#ffffff; border:1px solid var(--line); border-radius:12px; padding:18px 20px; box-shadow:0 2px 10px rgba(17,39,64,.035);}
[data-testid="stMetricLabel"] {color:#5d7084; font-weight:550;}
[data-testid="stMetricValue"] {color:var(--ink); font-weight:650; letter-spacing:-.04em;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:6px; border-bottom:1px solid var(--line); margin-bottom:1rem;}
[data-testid="stTabs"] button {font-weight:600; font-size:.88rem; color:#52657b; padding:12px 16px; border-radius:8px 8px 0 0;}
[data-testid="stTabs"] button[aria-selected="true"] {color:#17486c; background:#eaf1f7;}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;background:white;padding:6px;}
.stButton>button[kind="primary"], [data-testid="stDownloadButton"] button {background:#153d5e; color:white; border:1px solid #153d5e; border-radius:8px; font-weight:600;}
.stButton>button[kind="primary"]:hover, [data-testid="stDownloadButton"] button:hover {background:#205a84; color:white;}
[data-baseweb="select"]>div, [data-testid="stTextInput"] input {border-radius:8px;}
[data-testid="stFileUploader"] {background:white;border:1px solid var(--line);border-radius:12px;padding:12px;}
[data-testid="stAlert"] {border-radius:10px;}
hr {border-color:var(--line);}
.cvs-eyebrow {font-size:.73rem;letter-spacing:.13em;text-transform:uppercase;color:#397194;font-weight:700;margin-bottom:.25rem;}
.cvs-divider {height:1px;background:#dce5ed;margin:1rem 0 1.4rem;}
.cvs-footer {font-size:.8rem;color:#718298;}
@media(max-width:800px){.block-container{padding:1.1rem .9rem 2rem;}[data-testid="stTabs"] button{padding:9px 7px;font-size:.75rem;}}
/* Export and download buttons only */
div.stDownloadButton > button,
div.stDownloadButton > button *,
div[data-testid="stDownloadButton"] button,
div[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Keep export button text white on hover */
div.stDownloadButton > button:hover,
div.stDownloadButton > button:hover *,
div[data-testid="stDownloadButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
</style>""", unsafe_allow_html=True)
px.defaults.template='plotly_white'
px.defaults.color_discrete_sequence=['#1c5278','#4283aa','#7899b6','#9d7c58','#667e75']
px.defaults.color_continuous_scale=['#e9f1f7','#8eb3cc','#1b567d']
TARGET_DATE=pd.Timestamp('2027-01-01')
FEATURES_NUM=['living_area','bedrooms','bathrooms','effective_year_built','actual_year_built']
FEATURES_CAT=['abs_subdv_cd','hood_cd','quality_cd','map_id']
YEAR_FIELDS = {'effective_year_built', 'actual_year_built'}
YEAR_COLUMN_CONFIG = {
    'effective_year_built': st.column_config.NumberColumn('Effective year built', format='%d'),
    'actual_year_built': st.column_config.NumberColumn('Actual year built', format='%d'),
}

def format_property_input(field, value):
    if pd.isna(value):
        return 'Missing'
    if field in YEAR_FIELDS:
        return f'{float(value):.0f}'
    return str(value)

MODEL_NAMES=['Ridge Regression','Random Forest','Model G (HistGradientBoosting)','CatBoost','XGBoost']
MODEL_SHORT_DESCRIPTIONS = {
    'Ridge Regression': 'Linear equation with restrained adjustments for property features.',
    'Random Forest': 'Averages many decision trees to capture property differences.',
    'Model G (HistGradientBoosting)': 'Adds trees in stages to correct earlier prediction errors.',
    'CatBoost': 'Builds trees in stages to learn combinations of property features.',
    'XGBoost': 'Boosted trees with sampling and penalties to limit overfitting.',
}

def format_model_option(name):
    return f'{name} — {MODEL_SHORT_DESCRIPTIONS[name]}'

ALIASES={'Quality Code':'quality_cd','Quality code':'quality_cd','effective_yr_blt':'effective_year_built','actual_year':'actual_year_built','sl_price':'sale_price','sl_dt':'sale_date','market':'market_value','Living Area Sq Ft':'living_area','Sale Living Area Sq Ft':'living_area','Bedrooms':'bedrooms','Bathrooms':'bathrooms','Effective Year Built':'effective_year_built','Actual Year Built':'actual_year_built','Latitude':'latitude','Longitude':'longitude','current_market_value':'market_value','prop_val_yr':'valuation_year'}

def normalize(raw):
    df=raw.rename(columns={k:v for k,v in ALIASES.items() if k in raw and v not in raw}).copy()
    for c in FEATURES_NUM+['sale_price','market_value','latitude','longitude','valuation_year']:
        if c in df:
            if c in ('bedrooms','bathrooms'):
                df[c]=df[c].astype('string').str.extract(r'^\s*(\d+(?:\.\d+)?)')[0]
            df[c]=pd.to_numeric(df[c],errors='coerce')
    for c in YEAR_FIELDS:
        if c in df: df[c]=df[c].round().astype('Int64')
    for c in FEATURES_CAT:
        if c in df:
            df[c]=df[c].astype('string').str.strip().replace({'':pd.NA,'NULL':pd.NA,'nan':pd.NA})
            if c=='quality_cd': df[c]=df[c].str.replace(r'(?<=\d)\.0+$','',regex=True)
    if 'sale_date' in df:df['sale_date']=pd.to_datetime(df.sale_date,errors='coerce')
    if 'prop_id' in df:df['prop_id']=df.prop_id.astype('string').str.strip()
    return df

def assess_input(raw):
    df=normalize(raw)
    req=['prop_id','sale_date','sale_price','living_area']
    missing=[c for c in req if c not in df]
    if missing:return None,None,'Missing required columns: '+', '.join(missing)
    df['reason']=''
    def flag(mask,reason):df.loc[mask,'reason']=df.loc[mask,'reason']+reason+'; '
    flag(df.prop_id.isna()|(df.prop_id==''),'Missing property ID')
    flag(df.sale_date.isna(),'Missing sale date')
    flag(df.sale_price.isna()|(df.sale_price<=0),'Nonpositive/missing sale price')
    flag(df.living_area.isna()|(df.living_area<=0),'Nonpositive/missing living area')
    flag(df.sale_date>=TARGET_DATE,'Sale date on/after valuation date')
    if 'DOR RPTC' in df:
        rptc=df['DOR RPTC'].astype('string').str.strip().str.zfill(2)
        flag(~rptc.isin(['01','02']),'DOR RPTC outside 01/02')
    if 'prop_type_cd' in df:flag(df.prop_type_cd.astype('string').str.strip()!='R','Non-real property')
    if 'property_use_cd' in df:flag(~df.property_use_cd.astype('string').str.strip().str.lstrip('0').isin(['400']),'Non-condominium property use')
    # Exact transaction duplicates only; repeated distinct sales remain eligible.
    dupe=df.duplicated(subset=['prop_id','sale_date','sale_price'],keep='first')
    flag(dupe,'Duplicate transaction')
    eligible=df[df.reason==''].copy()
    eligible['sale_year']=eligible.sale_date.dt.year
    return eligible,df[df.reason!=''].copy(),None

class RoomConstrainedRidge(Ridge):
    """Ridge objective with nonnegative bedroom/bathroom slopes only."""
    def fit(self, X, y):
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float)
        center=X.mean(axis=0); target=y.mean(); n=X.shape[1]
        lower=np.full(n,-np.inf); lower[1:3]=0
        solution=lsq_linear(np.vstack([X-center,np.sqrt(self.alpha)*np.eye(n)]),
            np.r_[y-target,np.zeros(n)],bounds=(lower,np.full(n,np.inf)),tol=1e-10)
        if not solution.success: raise ValueError('Constrained Ridge fit failed: '+solution.message)
        self.coef_=solution.x; self.intercept_=float(target-center@self.coef_)
        self.n_features_in_=n
        return self

class RoomConstrainedPipeline(Pipeline):
    def fit(self, X, y, **params):
        transformed=self.named_steps['pre'].fit_transform(X,y)
        constraints=[0,1,1]+[0]*(transformed.shape[1]-3)
        model=self.named_steps['model']
        if isinstance(model, CatBoostAdapter): model.constraints=constraints
        elif not isinstance(model, RoomConstrainedRidge):
            key='monotone_constraints' if model.__class__.__name__=='XGBRegressor' else 'monotonic_cst'
            model.set_params(**{key:tuple(constraints) if key=='monotone_constraints' else constraints})
        model.fit(transformed,y)
        return self

class CatBoostAdapter(RegressorMixin, BaseEstimator):
    """Avoid version-specific CatBoost/sklearn pipeline tag incompatibility."""
    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.estimator_=CatBoostRegressor(iterations=450,depth=5,learning_rate=.045,loss_function='RMSE',verbose=False,random_seed=42,thread_count=2,
            monotone_constraints=self.constraints)
        self.estimator_.fit(X,y)
        return self
    def predict(self,X):
        return self.estimator_.predict(X)

def model_factory(name):
    # Codes are categories, never a numeric ranking. Dense output supports all five estimators.
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]) if name=='Ridge Regression' else SimpleImputer(strategy='median',keep_empty_features=True)
    pre=ColumnTransformer([('num',numeric,FEATURES_NUM),('cat',Pipeline([
        ('impute',SimpleImputer(strategy='constant',fill_value='Missing')),
        ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),FEATURES_CAT)],sparse_threshold=0)
    if name=='Ridge Regression': model=RoomConstrainedRidge(alpha=25.0)
    elif name=='Random Forest': model=RandomForestRegressor(n_estimators=180,min_samples_leaf=3,max_features=0.85,n_jobs=-1,random_state=42)
    elif name=='Model G (HistGradientBoosting)': model=HistGradientBoostingRegressor(max_iter=230,learning_rate=.045,max_leaf_nodes=18,l2_regularization=8,random_state=42)
    elif name=='CatBoost': model=CatBoostAdapter()
    elif name=='XGBoost':
        from xgboost import XGBRegressor
        model=XGBRegressor(n_estimators=420,max_depth=4,learning_rate=.045,subsample=.85,colsample_bytree=.9,reg_lambda=12,objective='reg:squarederror',n_jobs=2,random_state=42)
    else: raise ValueError(name)
    return RoomConstrainedPipeline([('pre',pre),('model',model)])

def validation_split(eligible):
    # Random property groups across ALL eligible sale years; repeated parcels cannot leak.
    split=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
    train_index,test_index=next(split.split(eligible,groups=eligible.prop_id))
    return eligible.iloc[train_index].copy(),eligible.iloc[test_index].copy()

def official_prb(y, p):
    """IAAO-style PRB: OLS of relative ratio deviation on log2 value proxy.

    The value proxy is (sale price + predicted value / median ratio) / 2.
    Returns the coefficient; uses a common sample and omits invalid observations.
    """
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    valid=np.isfinite(y)&np.isfinite(p)&(y>0)&(p>0)
    y=y[valid]; p=p[valid]
    if len(y)<10:return np.nan
    ratios=p/y
    median=float(np.median(ratios))
    if not np.isfinite(median) or median<=0:return np.nan
    value_proxy=(y+p/median)/2.0
    valid=(value_proxy>0)&np.isfinite(value_proxy)
    x=np.log2(value_proxy[valid]); dependent=ratios[valid]/median-1.0
    if len(x)<10 or np.ptp(x)<1e-10:return np.nan
    return float(np.polyfit(x,dependent,1)[0])

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    mask=np.isfinite(y)&np.isfinite(p)&(y>0)&(p>0)
    y=y[mask];p=p[mask]
    if not len(y):return {'# of Sales':0,'MAE':np.nan,'MAPE (%)':np.nan,'R²':np.nan,'Median ratio':np.nan,'COD (%)':np.nan,'PRD':np.nan,'PRB':np.nan}
    ratios=p/y
    median=np.median(ratios)
    cod=100*np.mean(np.abs(ratios-median))/median if median>0 else np.nan
    prd=np.mean(ratios)/(np.sum(p)/np.sum(y)) if np.sum(y)>0 and np.sum(p)>0 else np.nan
    return {'# of Sales':len(y),'MAE':mean_absolute_error(y,p),'MAPE (%)':100*np.mean(np.abs(p-y)/y),'R²':r2_score(y,p) if len(y)>1 else np.nan,'Median ratio':median,'COD (%)':cod,'PRD':prd,'PRB':official_prb(y,p)}

def prepare_features(df):
    out=df.copy()
    for c in FEATURES_NUM:
        if c not in out:out[c]=np.nan
        out[c]=pd.to_numeric(out[c],errors='coerce')
        if out[c].isna().all():out[c]=0.0
    for c in FEATURES_CAT:
        if c not in out:out[c]='Missing'
        out[c]=out[c].astype('object').where(out[c].notna(),'Missing').astype(str)
    return out[FEATURES_NUM+FEATURES_CAT]


@st.cache_resource(show_spinner=False)
def fit_explanation_model(data_bytes, name):
    eligible, _, error = assess_input(pd.read_csv(io.BytesIO(data_bytes), low_memory=False))
    if error:
        raise ValueError(error)
    training, _ = validation_split(eligible)
    Xtraining = prepare_features(training)
    fitted = model_factory(name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        fitted.fit(Xtraining, training.sale_price)
    return fitted, Xtraining

@st.cache_data(show_spinner=False)
def property_contributions(data_bytes, name, position):
    import math
    fitted, Xtraining = fit_explanation_model(data_bytes, name)
    eligible, _, error = assess_input(pd.read_csv(io.BytesIO(data_bytes), low_memory=False))
    if error:
        raise ValueError(error)
    # Prepare the entire holdout exactly as evaluate() does, THEN select a row.
    _, holdout = validation_split(eligible)
    target = prepare_features(holdout).iloc[[position]]
    background = Xtraining.sample(n=min(32, len(Xtraining)), random_state=42).reset_index(drop=True)
    fields = list(target.columns)
    n = len(fields)
    batches = []
    for mask in range(1 << n):
        hybrid = background.copy()
        for j, field in enumerate(fields):
            if mask & (1 << j):
                hybrid[field] = target.iloc[0][field]
        batches.append(hybrid)
    values = np.maximum(fitted.predict(pd.concat(batches, ignore_index=True)), 1).reshape(1 << n, len(background)).mean(axis=1)
    effects = np.zeros(n)
    for j in range(n):
        for mask in range(1 << n):
            if mask & (1 << j):
                continue
            k = mask.bit_count()
            effects[j] += (values[mask | (1 << j)] - values[mask]) / (n * math.comb(n-1, k))
    baseline, predicted = float(values[0]), float(values[-1])
    if not np.isclose(baseline + effects.sum(), predicted, rtol=1e-9, atol=.01):
        raise ValueError('Contribution reconciliation failed.')
    return fields, effects, baseline, predicted, len(background)

FEATURE_LABELS = {
    'living_area':'Living area', 'bedrooms':'Bedrooms', 'bathrooms':'Bathrooms',
    'effective_year_built':'Effective year built', 'actual_year_built':'Actual year built',
    'abs_subdv_cd':'Subdivision', 'hood_cd':'Neighborhood', 'quality_cd':'Quality code', 'map_id':'Map ID'
}

def horizontal_property_waterfall(fields, effects, baseline, predicted, property_inputs):
    import html
    import plotly.graph_objects as go
    order = np.argsort(-np.abs(effects), kind='stable')
    contributions = np.asarray(effects)[order]
    sorted_fields = [fields[j] for j in order]
    n = len(fields)
    starts, ends = np.empty(n), np.empty(n)
    level = baseline
    for j in range(n-1, -1, -1):
        starts[j] = level
        level += contributions[j]
        ends[j] = level
    span = max(float(np.ptp(np.r_[starts, ends, baseline, predicted])), 1000.0)
    pad = span * .14
    left = min(starts.min(), ends.min(), baseline, predicted) - pad
    right = max(starts.max(), ends.max(), baseline, predicted) + pad
    fig = go.Figure()
    labels = []
    for j, field in enumerate(sorted_fields):
        value = property_inputs[field]
        shown = 'Missing (model preprocessing applies)' if pd.isna(value) else format_property_input(field, value)
        if field in FEATURES_NUM and field not in YEAR_FIELDS and pd.notna(value):
            shown = f'{float(value):,.0f}' if float(value).is_integer() else f'{float(value):,.2f}'
            if field == 'living_area': shown += ' SF'
        labels.append(f'<span style="color:#888">{html.escape(shown)} = </span>{FEATURE_LABELS[field]}')
        start, end, delta = starts[j], ends[j], contributions[j]
        color = '#ff0051' if delta > 0 else '#008bfb'
        head = min(span * .016, abs(delta) * .45)
        shoulder = end-head if delta >= 0 else end+head
        if abs(delta) < 1e-8:
            fig.add_trace(go.Scatter(x=[start], y=[j], mode='markers', marker=dict(color='#999',size=5), hovertemplate=f'{FEATURE_LABELS[field]}: $0<extra></extra>',showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=[start,shoulder,end,shoulder,start,start], y=[j-.32,j-.32,j,j+.32,j+.32,j-.32],
                mode='lines',fill='toself',fillcolor=color,line=dict(color=color,width=.5),
                text=f'{FEATURE_LABELS[field]}<br>Input: {html.escape(shown)}<br>Contribution: {delta:+,.2f}<br>Running value: ${end:,.2f}',
                hovertemplate='%{text}<extra></extra>',showlegend=False))
        inside = abs(delta) > span * .20
        x = (start+end)/2 if inside else (max(start,end)+span*.012 if delta>=0 else min(start,end)-span*.012)
        fig.add_annotation(x=x,y=j,text=f'{"+" if delta>0 else "−" if delta<0 else ""}${abs(delta):,.0f}',
            showarrow=False,font=dict(color='white' if inside else color,size=14),
            xanchor='center' if inside else 'left' if delta>=0 else 'right')
        if j < n-1:
            fig.add_shape(type='line',x0=start,x1=start,y0=j+.33,y1=j+1-.33,line=dict(color='#aaa',width=1,dash='dot'))
    for value in [baseline,predicted]:
        fig.add_shape(type='line',x0=value,x1=value,y0=-.62,y1=n-.42,line=dict(color='#999',width=1,dash='dash'),layer='below')
    fig.add_annotation(x=predicted,y=-.91,text=f'f(x) = ${predicted:,.0f}',showarrow=False,font=dict(size=15,color='#333'))
    fig.add_annotation(x=baseline,y=-.15,xref='x',yref='paper',text=f'Reference E[f(X)] = ${baseline:,.0f}',showarrow=False,font=dict(size=13,color='#777'))
    fig.update_layout(title='Characteristic contributions to this property estimate',height=650,
        margin=dict(l=330,r=100,t=90,b=100),paper_bgcolor='white',plot_bgcolor='white',
        xaxis=dict(range=[left,right],tickprefix='$',tickformat=',.0f',title=None,showgrid=False,zeroline=False),
        yaxis=dict(tickmode='array',tickvals=list(range(n)),ticktext=labels,range=[n-.4,-1.35],
            gridcolor='#eee',griddash='dot',zeroline=False,tickfont=dict(size=13)))
    return fig


@st.cache_data(show_spinner=False)
def evaluate(file_bytes,model_names):
    raw=pd.read_csv(io.BytesIO(file_bytes),low_memory=False)
    good,excluded,error=assess_input(raw)
    if error:return None,None,None,None,error
    if good.prop_id.nunique()<2:return good,excluded,None,None,'Need at least two distinct properties for random validation.'
    train,test=validation_split(good)
    if len(train)<50 or len(test)<20:return good,excluded,None,None,'Need at least 50 training sales and 20 holdout sales after the random property split.'
    Xtr=prepare_features(train);Xte=prepare_features(test)
    results=[];preds=test.copy();failures={}
    for name in model_names:
        try:
            m, _ = fit_explanation_model(file_bytes, name)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',category=FutureWarning)
                pred=np.maximum(m.predict(Xte),1)
            preds[name]=pred
            results.append({'Model':name,**metrics(test.sale_price,pred)})
        except Exception as e:failures[name]=str(e)
    return good,excluded,preds,pd.DataFrame(results),failures

def map_property_key(series):
    return series.astype('string').str.strip().str.replace(r'\.0+$', '', regex=True)


def map_group_keys(series):
    labels = series.astype('string').str.strip().replace({'': pd.NA, 'NULL': pd.NA, 'nan': pd.NA, 'Missing': pd.NA})
    numeric = pd.to_numeric(labels, errors='coerce')
    return labels.where(numeric.isna(), numeric.map(lambda v: format(v, '.12g') if pd.notna(v) else pd.NA))


def attach_map_metric(mapped, validation, model_name, metric_name, group_column='abs_subdv_cd'):
    out = mapped.copy()
    out['Map_Metric'] = np.nan
    out['Validation_Sales_N'] = 0
    if group_column not in out or group_column not in validation or model_name not in validation:
        return out
    validation = validation.copy()
    validation['_map_group'] = map_group_keys(validation[group_column])
    out['_map_group'] = map_group_keys(out[group_column])
    records = []
    for group, rows in validation.dropna(subset=['_map_group']).groupby('_map_group'):
        result = metrics(rows.sale_price, rows[model_name])
        value = result[metric_name] if result['# of Sales'] >= 3 else np.nan
        records.append({'_map_group': group, 'Map_Metric': value, 'Validation_Sales_N': result['# of Sales']})
    if records:
        out = out.drop(columns=['Map_Metric','Validation_Sales_N']).merge(pd.DataFrame(records), on='_map_group', how='left', validate='many_to_one')
        out['Validation_Sales_N'] = out.Validation_Sales_N.fillna(0).astype(int)
    return out.drop(columns='_map_group')


@st.cache_data(show_spinner=False)
def inventory_map_predictions(sales_bytes, inventory_bytes, model_name):
    history = pd.read_csv(io.BytesIO(sales_bytes), low_memory=False)
    eligible, _, error = assess_input(history)
    if error:
        raise ValueError(error)
    if len(eligible) < 50:
        raise ValueError('Need at least 50 eligible historical sales to refit for inventory scoring.')
    inventory = normalize(pd.read_csv(io.BytesIO(inventory_bytes), dtype={'prop_id':'string'}, low_memory=False))
    required = ['prop_id'] + FEATURES_NUM + FEATURES_CAT
    missing = [col for col in required if col not in inventory]
    if missing:
        raise ValueError('Inventory is missing predictor columns: ' + ', '.join(missing))
    inventory['prop_id'] = map_property_key(inventory.prop_id)
    if inventory.prop_id.isna().any() or inventory.prop_id.eq('').any():
        raise ValueError('Inventory contains missing property IDs.')
    if inventory.prop_id.duplicated().any():
        raise ValueError('Inventory must contain one row per prop_id; resolve duplicate/supplement rows first.')
    scoreable = inventory.living_area.notna() & inventory.living_area.gt(0)
    if not scoreable.any():
        raise ValueError('No inventory properties have positive living area.')
    if 'property_use_cd' in inventory:
        if not pd.to_numeric(inventory.property_use_cd, errors='coerce').eq(400).all():
            raise ValueError('Inventory includes non-condominium property use codes.')
    if 'prop_type_cd' in inventory:
        if not inventory.prop_type_cd.astype('string').str.strip().eq('R').all():
            raise ValueError('Inventory includes non-real property records.')
    model = model_factory(model_name)
    model.fit(prepare_features(eligible), eligible.sale_price)
    inventory['map_value'] = np.nan
    inventory['Prediction_Status'] = np.where(scoreable, 'Estimated', 'Excluded: missing/nonpositive living area')
    inventory.loc[scoreable, 'map_value'] = np.maximum(model.predict(prepare_features(inventory.loc[scoreable])), 1)
    return inventory, len(eligible)


st.markdown('<div class="cvs-eyebrow">Assessment analytics / model research</div>',unsafe_allow_html=True)
st.title('Condo Valuation Studio')
st.markdown('<div class="cvs-divider"></div>',unsafe_allow_html=True)
st.caption('2027 Mass Appraisal Research  |  Model Validation  /  Equity Diagnostics  /  Property Review')
APP_DIR = Path(__file__).resolve().parent
SALES_DEFAULT = APP_DIR / 'Revised_Condo_Sales_v3_With_Coordinates.csv'
INVENTORY_DEFAULT = APP_DIR / '2027_ALL_Condos_v2_With_Coordinates.csv'
def select_dataset_bytes(mode, default_path, upload):
    if mode == 'Preloaded dataset':
        if not default_path.is_file():
            raise FileNotFoundError(f'Preloaded file missing: {default_path.name}. Choose Upload different dataset.')
        return default_path.read_bytes(), default_path.name
    if mode == 'Upload different dataset' and upload is not None:
        return upload.getvalue(), upload.name
    return None, None


with st.sidebar:
    st.header('Workspace')
    st.caption('DATASETS')
    st.subheader('Historical condo sales')
    sales_mode = st.radio('Sales data source', ['Preloaded dataset', 'Upload different dataset', 'Remove dataset'], key='sales_data_source')
    uploaded = st.file_uploader('Choose historical sales CSV', type=['csv'], key='sales_replacement') if sales_mode == 'Upload different dataset' else None
    st.subheader('Condo inventory')
    inventory_mode = st.radio('Inventory data source', ['Preloaded dataset', 'Upload different dataset', 'Remove dataset'], key='inventory_data_source')
    inventory_upload = st.file_uploader('Choose condo inventory CSV', type=['csv'], key='inventory_replacement') if inventory_mode == 'Upload different dataset' else None
    st.caption('Remove dataset clears it from this session. Your saved CSV is retained; choose Preloaded dataset to restore it.')

try:
    file_bytes, sales_source = select_dataset_bytes(sales_mode, SALES_DEFAULT, uploaded)
except FileNotFoundError as error:
    st.sidebar.error(str(error))
    file_bytes, sales_source = None, None
try:
    inventory_bytes, inventory_source = select_dataset_bytes(inventory_mode, INVENTORY_DEFAULT, inventory_upload)
except FileNotFoundError as error:
    st.sidebar.error(str(error))
    inventory_bytes, inventory_source = None, None
if file_bytes is not None:
    st.sidebar.success('Sales loaded: ' + sales_source)
else:
    st.sidebar.info('No historical sales dataset selected.')
if inventory_bytes is not None:
    st.sidebar.success('Inventory loaded: ' + inventory_source)
else:
    st.sidebar.info('No inventory dataset selected.')
if file_bytes is None:
    st.info('Select preloaded historical sales or upload a sales CSV in the sidebar to run the dashboard.')
    st.stop()
raw=pd.read_csv(io.BytesIO(file_bytes),low_memory=False)
good,excluded,validation_error=assess_input(raw)
if validation_error:st.error(validation_error);st.stop()
with st.sidebar:
    st.success(f'{len(good):,} qualified sales')
    st.caption(f'{len(excluded):,} excluded records')
    chosen=st.multiselect('Models to evaluate',MODEL_NAMES,default=MODEL_NAMES)
    st.caption('Validation randomly assigns 80% of properties to training and 20% to testing across all eligible sale years (seed 42). All sales for a property stay together.')
if not chosen:st.warning('Select at least one model.');st.stop()
with st.spinner('Training and evaluating models on a shared random holdout...'):
    good,excluded,preds,comparison,problems=evaluate(file_bytes,tuple(chosen))
if isinstance(problems,str):st.error(problems);st.stop()
if problems:
    with st.expander('Model warnings / failures'):
        for k,v in problems.items():st.warning(f'{k}: {v}')
if comparison is None or comparison.empty:st.error('No model trained successfully. Check model warnings.');st.stop()
working=comparison.Model.tolist()
tabs=st.tabs(['Data Overview','Individual Models','Model Comparison','Stratification Lab','Property Review','Sales & Prediction Map'])
with tabs[0]:
    st.subheader('Dataset qualification')
    a,b,c,d=st.columns(4)
    a.metric('Uploaded rows',f'{len(raw):,}');b.metric('Qualified sales',f'{len(good):,}');c.metric('Unique properties',f'{good.prop_id.nunique():,}');d.metric('Excluded rows',f'{len(excluded):,}')
    st.dataframe(good.sale_year.value_counts().sort_index().rename_axis('Sale year').reset_index(name='Sales'),hide_index=True)
    st.markdown('**Fields detected**');st.write(', '.join(good.columns))
    if len(excluded):
        st.dataframe(excluded[['prop_id','reason']].head(200),hide_index=True)
        st.download_button('Download exclusion report',excluded.to_csv(index=False).encode(),'excluded_sales.csv','text/csv')
    st.caption('Only DOR RPTC 01/02 are included when that field exists. The original upload is never overwritten.')
with tabs[1]:
    name=st.selectbox('Inspect model',working,key='inspect_model',format_func=format_model_option)
    m=comparison.set_index('Model').loc[name]
    # Consistent accuracy and equity diagnostics, including IAAO-style PRB.
    cols=st.columns(3)
    for col,(label,value,fmt) in zip(cols,[('MAE',m['MAE'],'${:,.0f}'),('MAPE',m['MAPE (%)'],'{:.2f}%'),('R²',m['R²'],'{:.3f}')]):
        col.metric(label,fmt.format(value))
    cols=st.columns(3)
    for col,(label,value,fmt) in zip(cols,[('COD',m['COD (%)'],'{:.2f}%'),('PRD',m['PRD'],'{:.3f}'),('PRB',m['PRB'],'{:.3f}')]):
        col.metric(label,fmt.format(value) if pd.notna(value) else 'N/A')
    st.caption('PRB uses the IAAO-style regression of (ratio / median ratio - 1) on log2[(sale price + predicted value / median ratio) / 2]. These are MODEL-prediction-to-sale equity diagnostics, not an official PACS assessment ratio study. PRB significance and confidence intervals are not yet reported.')
    view=preds[['prop_id','sale_price',name]].rename(columns={'sale_price':'Actual sale price',name:'Predicted sale price'})
    fig=px.scatter(view,x='Actual sale price',y='Predicted sale price',hover_data=['prop_id'],title='Predicted versus actual — random holdout',opacity=.65)
    hi=max(view['Actual sale price'].max(),view['Predicted sale price'].max());fig.add_shape(type='line',x0=0,y0=0,x1=hi,y1=hi,line=dict(color='#dc6c36',dash='dash'));st.plotly_chart(fig,use_container_width=True)
    errors=view.copy();errors['Error ($)']=errors['Predicted sale price']-errors['Actual sale price'];errors['Absolute % error']=100*errors['Error ($)'].abs()/errors['Actual sale price']
    st.plotly_chart(px.histogram(errors,x='Error ($)',nbins=35,title='Prediction error distribution'),use_container_width=True)
    st.dataframe(errors.sort_values('Absolute % error',ascending=False),hide_index=True)
    st.download_button('Download this model’s validation predictions',errors.to_csv(index=False).encode(),name.replace(' ','_')+'_holdout.csv','text/csv')
with tabs[2]:
    st.subheader('Compare models on the identical random holdout')
    st.dataframe(comparison.sort_values('MAPE (%)').style.format({'MAE':'${:,.0f}','MAPE (%)':'{:.2f}%','R²':'{:.3f}','COD (%)':'{:.2f}%','PRD':'{:.3f}','Median ratio':'{:.3f}','PRB':'{:.3f}'}),hide_index=True,use_container_width=True)
    metric_options=['MAE','MAPE (%)','R²','COD (%)','PRD','PRB','Median ratio']
    chart_options=['Vertical bar','Horizontal bar','Dot plot','Lollipop','Line with markers','Area','Heatmap','Box plot','Violin plot','ECDF']
    left,right=st.columns(2)
    with left: selected_metric=st.selectbox('Metric to compare',metric_options,index=1,key='compare_metric')
    with right: selected_chart=st.selectbox('Graph type',chart_options,key='compare_chart')
    # Distribution charts use per-sale observations. Aggregate scorecards use one value per model.
    score=comparison[['Model',selected_metric]].dropna().copy()
    score=score.sort_values(selected_metric,ascending=True)
    title=f'{selected_metric} comparison — random holdout'
    if selected_chart=='Vertical bar':
        fig=px.bar(score,x='Model',y=selected_metric,color='Model',title=title,text_auto='.3g')
    elif selected_chart=='Horizontal bar':
        fig=px.bar(score,x=selected_metric,y='Model',orientation='h',color='Model',title=title,text_auto='.3g')
    elif selected_chart=='Dot plot':
        fig=px.scatter(score,x='Model',y=selected_metric,color='Model',title=title)
        fig.update_traces(marker_size=13)
    elif selected_chart=='Lollipop':
        fig=px.scatter(score,x='Model',y=selected_metric,color='Model',title=title)
        fig.update_traces(marker_size=12)
        for _,r in score.iterrows():fig.add_shape(type='line',x0=r['Model'],x1=r['Model'],y0=0,y1=r[selected_metric],line=dict(color='#94a3b8',width=2))
    elif selected_chart=='Line with markers':
        fig=px.line(score,x='Model',y=selected_metric,markers=True,title=title)
        st.caption('Models are categories, not a time series; connecting lines are for visual comparison only.')
    elif selected_chart=='Area':
        fig=px.area(score,x='Model',y=selected_metric,title=title)
        st.caption('Models are categories, not a continuous scale; area does not represent a cumulative quantity.')
    elif selected_chart=='Heatmap':
        fig=px.imshow(score.set_index('Model')[[selected_metric]].T,aspect='auto',text_auto='.3g',color_continuous_scale='Blues',title=title)
    else:
        # Box/violin/ECDF are meaningful for distributions, so construct the
        # selected metric on bootstrap resamples of the shared holdout.
        rng=np.random.default_rng(42)
        rows=[]
        actual=preds.sale_price.to_numpy(dtype=float)
        n=len(actual)
        for model_name in working:
            predicted=preds[model_name].to_numpy(dtype=float)
            for iteration in range(150):
                indices=rng.integers(0,n,size=n)
                value=metrics(actual[indices],predicted[indices])[selected_metric]
                if np.isfinite(value):rows.append({'Model':model_name,selected_metric:value})
        samples=pd.DataFrame(rows)
        if samples.empty:
            st.warning('Not enough valid resamples for this metric and chart.')
            fig=None
        elif selected_chart=='Box plot':fig=px.box(samples,x='Model',y=selected_metric,color='Model',points=False,title=title+' (bootstrap variability)')
        elif selected_chart=='Violin plot':fig=px.violin(samples,x='Model',y=selected_metric,color='Model',box=True,points=False,title=title+' (bootstrap variability)')
        else:fig=px.ecdf(samples,x=selected_metric,color='Model',title=title+' (bootstrap cumulative distribution)')
        st.caption('Distribution charts show 150 bootstrap resamples of the SAME holdout sales per model, not 150 independent validation tests. Resampling does not account for repeated-property or temporal dependence.')
    if fig is not None:
        fig.update_layout(height=520,showlegend=selected_chart in ['Box plot','Violin plot','ECDF'],margin=dict(t=65,b=55))
        st.plotly_chart(fig,use_container_width=True)
    st.caption('Lower is better for MAE, MAPE and COD. R² is better when higher. PRD is ideally near 1; PRB is ideally near 0. Median ratio is a level diagnostic, not an error ranking. PRB uses model predictions, not official assessed values.')
    st.download_button('Download model scorecard',comparison.to_csv(index=False).encode(),'model_comparison.csv','text/csv')
with tabs[1]:
    st.divider()
    st.subheader('Why this property received this value')
    st.caption('Choose a model to update the waterfall automatically. Your selected transaction stays the same across models.')
    descriptions = {
        'Ridge Regression': 'One additive equation. Each numeric characteristic has a constant slope; category codes have separate offsets. Alpha 25 shrinks coefficients to restrain unstable adjustments. Numeric inputs are standardized before fitting.',
        'Random Forest': 'Averages 180 decision trees. Each tree divides comparable sales into groups using characteristics. Leaves have at least 3 training sales. Effects depend on the other characteristics; there is no fixed countywide dollar adjustment per feature.',
        'Model G (HistGradientBoosting)': 'Builds up to 230 successive trees, each correcting the current prediction errors. Learning rate 0.045, up to 18 leaves per tree, and L2 penalty 8 restrain complexity. This is the dashboard benchmark, not the legacy saved Model G.',
        'CatBoost': 'Builds 450 successive trees with depth 5 and learning rate 0.045. Category codes, including quality, are one-hot encoded before fitting.',
        'XGBoost': 'Builds 420 successive trees with depth 4 and learning rate 0.045, using row and feature subsampling plus L2 penalty 12. Category codes, including quality, are one-hot encoded before fitting.'
    }
    explain_name = st.selectbox('Model to explain', working, key='explain_model')
    st.write(descriptions[explain_name])
    st.caption('Validation model: the same random 80/20 property split across all eligible sale years (seed 42). Explanations use the validation training population, not the inventory refit.')
    st.write('Inputs: living area, bedrooms, bathrooms, effective year built, actual year built, subdivision, neighborhood, quality code, and map ID.')
    st.info('Sale price is the training target. Existing market value, property ID, coordinates and sale date are not predictors. No explicit time adjustment to January 1, 2027 or 85% ratio multiplier is applied.')
    st.caption('All models use quality as a category. Quality differences are learned from training sales and can be zero where evidence is limited. Bedrooms and bathrooms have nondecreasing effects when other inputs are fixed; a Shapley contribution relative to different reference properties can still be negative.')
    st.caption('Missing numeric values normally use training medians. A numeric column entirely missing within a prepared population is set to zero by the existing app. Missing categories use “Missing”; unseen categories are ignored by the one-hot encoder.')
    options = list(range(len(preds)))
    row_position = st.selectbox('Holdout transaction to explain', options,
        format_func=lambda j: f"{preds.iloc[j].prop_id} | {preds.iloc[j].sale_date:%Y-%m-%d} | ${preds.iloc[j].sale_price:,.0f}", key='explain_transaction')
    selected_row = preds.iloc[[row_position]]
    Xproperty = prepare_features(preds).iloc[[row_position]]
    st.dataframe(Xproperty, hide_index=True, use_container_width=True, column_config=YEAR_COLUMN_CONFIG)
    try:
        with st.spinner('Fitting validation model and calculating contributions...'):
            fields, effects, baseline, predicted, background_n = property_contributions(file_bytes, explain_name, row_position)
            fitted, Xtraining = fit_explanation_model(file_bytes, explain_name)
        import plotly.graph_objects as go
        a,b,c = st.columns(3)
        a.metric('Reference baseline', f'${baseline:,.0f}')
        b.metric('Explained estimate', f'${predicted:,.0f}')
        c.metric('Actual sale price', f'${selected_row.iloc[0].sale_price:,.0f}')
        if not np.isclose(predicted, float(selected_row.iloc[0][explain_name]), rtol=1e-7, atol=.01):
            raise ValueError('Explanation does not match the displayed validation prediction; the waterfall has been withheld.')
        fig = horizontal_property_waterfall(fields, effects, baseline, predicted, Xproperty.iloc[0])
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Pink pushes the estimate higher; blue pushes it lower. Contributions are ranked by absolute size. Read from the reference baseline at the bottom upward.')
        breakdown = pd.DataFrame({'Characteristic':[FEATURE_LABELS[f] for f in fields], 'Property input':[format_property_input(f, Xproperty.iloc[0][f]) for f in fields], 'Contribution ($)':effects})
        st.dataframe(breakdown, hide_index=True, use_container_width=True)
        st.caption(f'Exact Shapley allocation over all 512 combinations of the nine inputs, relative to {background_n} sampled training rows (seed 42). The background is sampled; values can change with a different reference population. Hybrid combinations can be unrealistic for correlated features. Contributions explain model behavior, not causal effects or independent appraisal adjustments. Predictions include the dashboard’s $1 floor.')
        st.write(f'Reconciliation: ${baseline:,.2f} + ${effects.sum():,.2f} = ${predicted:,.2f}.')
        export_contributions = breakdown.assign(Property_ID=str(selected_row.iloc[0].prop_id), Sale_Date=str(selected_row.iloc[0].sale_date), Model=explain_name, Baseline=baseline, Prediction=predicted, Reference_Rows=background_n, Reference_Seed=42)
        st.download_button('Download property contributions', export_contributions.to_csv(index=False).encode(), 'property_contributions.csv', 'text/csv')
        if explain_name == 'Ridge Regression':
            pre = fitted.named_steps['pre']; ridge = fitted.named_steps['model']
            names = pre.get_feature_names_out()
            st.markdown('**Actual fitted Ridge coefficients**')
            st.caption('Numeric coefficients below are per one original input unit, holding all other inputs fixed. Category coefficients are offsets from the fitted intercept in this full one-hot parameterization; a category-to-category change equals the difference between their offsets. Correlated inputs can make individual coefficients unstable.')
            coefficients = np.asarray(ridge.coef_).copy()
            scaler = pre.named_transformers_['num'].named_steps['scale']
            coefficients[:len(FEATURES_NUM)] /= scaler.scale_
            original_intercept = float(ridge.intercept_ - np.sum(ridge.coef_[:len(FEATURES_NUM)] * scaler.mean_ / scaler.scale_))
            st.write(f'Intercept in original numeric units: ${original_intercept:,.2f}')
            st.dataframe(pd.DataFrame({'Term':names, 'Coefficient ($)':coefficients}), hide_index=True)
        st.markdown('**Change one characteristic**')
        change_field = st.selectbox('Characteristic', fields, key='explain_change_field')
        scenario = Xproperty.copy()
        if change_field in FEATURES_NUM:
            original = float(Xproperty.iloc[0][change_field])
            if not np.isfinite(original): original = float(Xtraining[change_field].median())
            replacement = st.number_input('Comparison year' if change_field in YEAR_FIELDS else 'Comparison value', value=original, step=1.0 if change_field in YEAR_FIELDS else None, format='%.0f' if change_field in YEAR_FIELDS else None, key=f'explain_numeric_{row_position}_{change_field}')
            scenario[change_field] = replacement
        else:
            choices = sorted(set(Xtraining[change_field].astype(str)) | {str(Xproperty.iloc[0][change_field])})
            replacement = st.selectbox('Comparison code', choices, index=choices.index(str(Xproperty.iloc[0][change_field])), key=f'explain_category_{row_position}_{change_field}')
            scenario[change_field] = replacement
        alternative = float(np.maximum(fitted.predict(scenario), 1)[0])
        st.metric('Comparison estimate', f'${alternative:,.0f}', delta=f'${alternative-predicted:,.0f}')
        if change_field in FEATURES_NUM:
            low, high = Xtraining[change_field].min(), Xtraining[change_field].max()
            if replacement < low or replacement > high: st.warning('Comparison input is outside the observed training range.')
        st.caption('Only this input changes. This is model sensitivity, not proof of a market adjustment. Changing living area without changing beds/baths, for example, may describe an unusual property.')
    except Exception as error:
        st.error(f'Explanation could not be calculated: {error}')

with tabs[3]:
    st.subheader('Stratification Lab')
    chosen_strat=st.multiselect('Compare models',working,default=working,key='strat_models')
    filter_cols=st.columns(3)
    with filter_cols[0]:
        years=st.multiselect('Sale years',sorted(preds.sale_year.dropna().unique().tolist()),default=sorted(preds.sale_year.dropna().unique().tolist()))
    with filter_cols[1]:
        subdivisions=sorted(preds.abs_subdv_cd.dropna().unique().tolist()) if 'abs_subdv_cd' in preds else []
        selected_sub=st.multiselect('Filter subdivisions (optional)',subdivisions)
    with filter_cols[2]:
        min_n=st.slider('Minimum holdout sales per group',3,30,5)
    dimensions={'Subdivision':'abs_subdv_cd','Neighborhood':'hood_cd','Quality code':'quality_cd','Sale year':'sale_year','Bedrooms':'bedrooms','Bathrooms':'bathrooms','Effective year built':'effective_year_built'}
    dim=st.selectbox('Group results by',list(dimensions))
    metric=st.selectbox('Comparison metric',['MAPE (%)','MAE','COD (%)','PRD','PRB','R²','Median ratio'])
    frame=preds[preds.sale_year.isin(years)].copy()
    if selected_sub:frame=frame[frame.abs_subdv_cd.isin(selected_sub)]
    group_col=dimensions[dim]
    if group_col not in frame:st.warning(f'{dim} not available in this dataset.')
    elif not chosen_strat:st.warning('Choose at least one model.')
    else:
        records=[]
        for group,sub in frame.groupby(group_col,dropna=False):
            if len(sub)<min_n:continue
            for name in chosen_strat:
                records.append({'Group':format_property_input(group_col, group),'Model':name,**metrics(sub.sale_price,sub[name])})
        out=pd.DataFrame(records)
        if out.empty:st.info('No strata meet the current minimum sample size.')
        else:
            wide=out.pivot(index='Group',columns='Model',values=metric).reset_index()
            st.dataframe(wide,hide_index=True,use_container_width=True)
            st.plotly_chart(px.bar(out,x='Group',y=metric,color='Model',barmode='group',hover_data=['# of Sales'],title=f'{metric} by {dim}'),use_container_width=True)
            st.download_button('Download stratified model results',out.to_csv(index=False).encode(),'stratified_model_results.csv','text/csv')
            st.caption('Small samples may be unstable. PRD and COD are diagnostics on predicted/sale ratios; Small-group PRB can be unstable and should be interpreted cautiously.')
with tabs[4]:
    st.subheader('Property-level validation review')
    props=sorted(preds.prop_id.dropna().unique().tolist())
    query=st.text_input('Search prop_id',placeholder='Enter full or partial property ID')
    matches=[p for p in props if query.lower() in p.lower()][:250] if query else props[:250]
    if not matches:st.warning('No matching holdout property ID.')
    else:
        prop=st.selectbox('Property ID',matches)
        sub=preds[preds.prop_id==prop].copy()
        cols=['prop_id','sale_date','sale_price','market_value','abs_subdv_cd','hood_cd','living_area','effective_year_built','actual_year_built','quality_cd']
        cols=[c for c in cols if c in sub]
        st.dataframe(sub[cols+working],hide_index=True,use_container_width=True,column_config=YEAR_COLUMN_CONFIG)
        subject=sub.iloc[0]
        peers=good.copy()
        if 'abs_subdv_cd' in peers and pd.notna(subject.get('abs_subdv_cd')):
            peers=peers[peers.abs_subdv_cd==subject.abs_subdv_cd]
            peer_label='Eligible sales in selected subdivision'
        else:
            peer_label='All eligible sales (subdivision unavailable)'
        peers=peers[peers.prop_id!=prop]
        if peers.empty:
            st.info('No other eligible comparable sales are available for a box plot.')
        else:
            import plotly.graph_objects as go
            fig=go.Figure(go.Box(y=peers.sale_price,name='Comparable sales',boxpoints='outliers',marker_color='#21918c'))
            for name in working:
                fig.add_trace(go.Scatter(x=['Comparable sales'],y=[subject[name]],mode='markers',name=name,marker=dict(size=11,symbol='diamond')))
            fig.add_trace(go.Scatter(x=['Comparable sales'],y=[subject.sale_price],mode='markers',name='Selected actual sale',marker=dict(size=14,symbol='star',color='#111111')))
            fig.update_layout(title='Selected property estimates versus comparable sale prices',yaxis=dict(title='Price',tickprefix='$',tickformat=',.0f'))
            st.plotly_chart(fig,use_container_width=True)
            st.caption(f'{peer_label}: {len(peers):,} sales across eligible years, excluding this property. Box = middle 50%; line = median; whiskers = 1.5 × IQR. Prices are not time-adjusted.')

    export=preds[['prop_id','sale_date','sale_price']+working].copy()
    st.download_button('Download all holdout property predictions',export.to_csv(index=False).encode(),'property_model_comparison.csv','text/csv')

with tabs[5]:
    st.subheader('Sales & Prediction Map')
    selection = st.selectbox('Choose sales year or prediction model',
        ['2024 sales', '2025 sales', '2026 sales'] + MODEL_NAMES, key='sales_prediction_map_choice')
    st.caption('Coordinates are current parcel/address-point locations. Multiple transactions at the same location overlap; download the table to inspect every sale.')
    mapped = None
    title = selection
    if selection.endswith(' sales'):
        year = int(selection[:4])
        mapped = good[good.sale_year.eq(year)].copy()
        mapped['map_value'] = mapped.sale_price
        value_label = 'Recorded sale price ($)'
        st.caption(f'{year} qualified historical sales; the same eligibility filters used by the dashboard apply.')
    else:
        value_label = 'Model estimate ($)'
        population = st.radio('Prediction population',
            ['Random holdout sales (validation)', 'Combined condo inventory (refit on all eligible sales)'],
            index=1 if inventory_bytes is not None else 0, key='map_prediction_population')
        if population.startswith('Random'):
            if selection in preds.columns:
                mapped = preds.copy()
                mapped['map_value'] = mapped[selection]
                title = selection + ' — random holdout'
                st.caption('Out-of-sample estimates for a random sample across all eligible sale years. Training excludes every sale for holdout property IDs. These are not 2027 inventory predictions.')
            else:
                st.warning('Select this model under “Models to evaluate” in the sidebar, then check model warnings if it fails to train.')
        elif inventory_bytes is None:
            st.info('Upload your all-condo inventory CSV in the sidebar to map inventory estimates.')
        else:
            st.caption('The selected model will refit on all eligible historical sales. Inventory estimates are not out-of-sample validation results and have no explicit market-time adjustment to January 1, 2027.')
            try:
                with st.spinner('Refitting selected model and scoring inventory...'):
                    mapped, training_count = inventory_map_predictions(file_bytes, inventory_bytes, selection)
                title = selection + ' — combined condo inventory'
                st.caption(f'Training sales: {training_count:,}. Inventory file: {inventory_source}.')
            except Exception as error:
                st.error(f'Inventory scoring failed: {error}')
    color_choice = st.selectbox('Color map by', ['Sale price / predicted value', 'MAE', 'MAPE', 'COD', 'PRD', 'PRB', 'R²'], key='map_color_variable')
    metric_columns = {'MAE':'MAE', 'MAPE':'MAPE (%)', 'COD':'COD (%)', 'PRD':'PRD', 'PRB':'PRB', 'R²':'R²'}
    if mapped is not None:
        color_field = 'map_value'
        color_label = value_label
        color_scale = ['#0057B8','#00A6A6','#FFD600','#FF7A00','#E60000']
        color_midpoint = None
        if color_choice in metric_columns:
            metric_model = selection if selection in MODEL_NAMES else st.selectbox('Model used for map diagnostics', working, key='map_diagnostic_model')
            if metric_model not in preds:
                st.warning('Enable this model in the sidebar to calculate its validation color metric.')
            mapped = attach_map_metric(mapped, preds, metric_model, metric_columns[color_choice])
            color_field = 'Map_Metric'
            color_label = color_choice + ' — subdivision / random holdout validation'
            mapped['Diagnostic_Model'] = metric_model
            st.caption(f'Color = {metric_model} subdivision-level {color_choice}, calculated from random holdout sales. Every property in the same subdivision shares the score. This is not an individual-property error or a score calculated from the refitted inventory model. For every sales-year layer, color describes the same random holdout validation performance.')
            st.caption('At least 3 valid holdout sales per subdivision; PRB requires at least 10 and variation in value. Missing subdivision scores are omitted and counted below.')
            if color_choice in ['MAE','MAPE','COD']:
                st.caption('Lower values indicate smaller errors or less ratio dispersion.')
            elif color_choice in ['PRD','PRB']:
                color_scale = ['#0057B8','#00A6A6','#FFD600','#FF7A00','#E60000']
                color_midpoint = 1.0 if color_choice == 'PRD' else 0.0
                st.caption('Reference: PRD = 1; PRB = 0. Blue/teal is below the reference; yellow marks the reference; orange/red is above. Both directions need review.')
            else:
                st.caption('Higher R² indicates better fit within the subdivision; values can be negative.')
        for col in ['latitude', 'longitude']:
            if col not in mapped:
                mapped[col] = np.nan
            mapped[col] = pd.to_numeric(mapped[col], errors='coerce')
        location_valid = mapped.latitude.between(29,31) & mapped.longitude.between(-84,-81)
        value_valid = pd.to_numeric(mapped.map_value, errors='coerce').gt(0)
        color_valid = pd.to_numeric(mapped[color_field], errors='coerce').notna()
        valid = location_valid & value_valid & color_valid
        if color_choice in metric_columns:
            st.caption(f'{int((location_valid & value_valid & ~color_valid).sum()):,} records have a location/value but no usable subdivision metric for this layer.')
        a,b,c = st.columns(3)
        a.metric('Selected records', f'{len(mapped):,}')
        b.metric('Mappable records', f'{int(valid.sum()):,}')
        c.metric('Missing / invalid location', f'{int((~location_valid).sum()):,}')
        if (~value_valid).any():
            st.warning(f'{int((~value_valid).sum()):,} properties have no estimate; see Prediction_Status in the table and download.')
        geo = mapped.loc[valid].copy()
        if geo.empty:
            st.info('No mappable records for this selection. Check Latitude, Longitude and prediction availability in the combined data file.')
        else:
            hover = {'latitude':False, 'longitude':False, 'map_value':':$,.0f'}
            for col in ['geo_id','Parcel_ID','sale_date','sale_price','abs_subdv_cd','hood_cd','living_area','quality_cd']:
                if col in geo and col != 'sale_price':
                    hover[col] = True
            if 'sale_price' in geo:
                hover['sale_price'] = ':$,.0f'
            if color_field == 'Map_Metric':
                hover['Map_Metric'] = ':,.3f'
                hover['Validation_Sales_N'] = True
                hover['Diagnostic_Model'] = True
            fig = px.scatter_map(geo, lat='latitude', lon='longitude', color=color_field,
                hover_name='prop_id', hover_data=hover,
                labels={'map_value':value_label, color_field:color_label},
                color_continuous_scale=color_scale, color_continuous_midpoint=color_midpoint,
                zoom=10, center={'lat':float(geo.latitude.median()),'lon':float(geo.longitude.median())},
                map_style='carto-positron', title=title)
            fig.update_traces(marker={'size':7, 'opacity':1.0})
            fig.update_layout(height=600, margin={'l':0,'r':0,'t':50,'b':0})
            st.plotly_chart(fig, use_container_width=True)
        mapped['Coordinate_Status_Map'] = np.select([~location_valid, ~value_valid, ~color_valid], ['Missing / invalid coordinates', 'Location available; no prediction', 'No usable subdivision metric'], default='Mapped')
        display_cols = ['prop_id','sale_date','sale_price','map_value','latitude','longitude','Coordinate_Status','Prediction_Status','Diagnostic_Model','Map_Metric','Validation_Sales_N','Coordinate_Status_Map']
        display_cols = [col for col in display_cols if col in mapped]
        st.dataframe(mapped[display_cols], hide_index=True, use_container_width=True,column_config=YEAR_COLUMN_CONFIG)
        st.download_button('Download selected map records', mapped.to_csv(index=False).encode(),
            'map_' + selection.replace(' ','_').replace('(','').replace(')','') + '.csv', 'text/csv', key='download_map_selection')
        if (~location_valid).any():
            st.download_button('Download records needing coordinates', mapped.loc[~location_valid].to_csv(index=False).encode(),
                'map_missing_coordinates.csv', 'text/csv', key='download_missing_map_locations')



st.divider()
st.caption('Research prototype v1.4 | Model G here is a newly trained histogram gradient boosting benchmark, NOT the separately saved legacy Model G artifact. Historical validation results are not 2027 roll predictions. The inventory map optionally refits on all eligible historical sales; it does not add a time adjustment to 2027.')
