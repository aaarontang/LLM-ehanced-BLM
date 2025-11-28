"""
Automated LLM Querying Module
"""
import os, json, re, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def load_prompt_template(path="message.txt"):
    if not os.path.exists(path): raise FileNotFoundError("Missing message.txt")
    with open(path, 'r', encoding='utf-8') as f: return f.read()

def prepare_single_stock_context(ticker, df):
    p_col, r_col = f'{ticker}_price', f'{ticker}_return'
    if p_col in df.columns and r_col in df.columns:
        p = df[p_col].dropna(); r = df[r_col].dropna()
        if not p.empty and not r.empty:
            r3m = (p.iloc[-1]/p.iloc[0]-1)*100
            vol = r.std() * np.sqrt(12) * 100
            return (f"\nTARGET: {ticker}\nSTATS: 3M_Ret={r3m:.1f}%, Vol={vol:.1f}%.\n"
                    f"COMMAND: 1. Search for specific catalysts (Earnings, Buybacks). "
                    f"2. IF NONE FOUND: Search for '{ticker} Sector Trends' and 'China Macro'. "
                    f"Form a view based on available data.")
    return None

def calculate_predicted_return(mu_daily, s_score, c_score, alpha, beta):
    # Additive Model
    trend = mu_daily * alpha
    # Sentiment direction (-1 to 1)
    signal = (s_score - 0.5) * 2.0 * beta * c_score
    return trend + signal

def extract_json(text):
    try: return json.loads(text)
    except: pass
    try: return json.loads(re.search(r"\{.*\}", text, re.DOTALL).group())
    except: return None

def query_llm_multiple(api_key, template, tickers, hist_df, n_queries=3, model="sonar-pro", alpha=0.5, beta=0.0015):
    if not OPENAI_AVAILABLE: return np.zeros(len(tickers)), np.eye(len(tickers)), {}
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    res_s, res_c = {t:[] for t in tickers}, {t:[] for t in tickers}
    
    def fetch(t, prompt):
        time.sleep(np.random.uniform(0.2, 0.6))
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role":"system","content":"JSON Only."}, {"role":"user","content":prompt}]
            )
            data = extract_json(resp.choices[0].message.content)
            if data:
                if t in data: return t, data[t]
                if 'S' in data: return t, data
            return t, None
        except: return t, None

    tasks = []
    for t in tickers:
        ctx = prepare_single_stock_context(t, hist_df)
        if ctx:
            full = template + "\n\n" + ctx
            for _ in range(n_queries): tasks.append((t, full))

    print(f"  > Processing {len(tasks)} queries...")
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(fetch, t, p) for t, p in tasks]
        for fut in as_completed(futures):
            t, d = fut.result()
            if d:
                res_s[t].append(np.clip(float(d.get('S',0.5)),0,1))
                res_c[t].append(np.clip(float(d.get('C',0.5)),0,1))

    Q, omega, conf = np.zeros(len(tickers)), np.zeros(len(tickers)), {}
    for i, t in enumerate(tickers):
        mu = hist_df[f'{t}_return'].mean()/21.0 if f'{t}_return' in hist_df else 0
        s, c = res_s[t], res_c[t]
        if s:
            preds = [calculate_predicted_return(mu, s[k], c[k], alpha, beta) for k in range(len(s))]
            Q[i] = np.mean(preds)
            omega[i] = np.var(preds) if len(preds)>1 else 1e-5
            conf[t] = np.mean(c)
        else:
            Q[i], omega[i], conf[t] = mu, 1e-3, 0.5
            
    return Q, np.diag(omega), {'raw_sentiments': res_s, 'raw_confidences': res_c, 'confidence': conf}

def generate_views_for_period(prices, returns, tickers, date, lookback_months=3, api_key=None, n_queries=3, model="sonar-pro", alpha=0.5, beta=0.0015):
    date = pd.to_datetime(date); start = date - pd.DateOffset(months=lookback_months)
    lb_p = prices.loc[(prices.index>=start)&(prices.index<date)]
    lb_r = returns.loc[(returns.index>=start)&(returns.index<date)]
    combined = pd.DataFrame(index=lb_p.index)
    for t in tickers:
        if t in lb_p: combined[f'{t}_price'] = lb_p[t]
        if t in lb_r: combined[f'{t}_return'] = lb_r[t]
    
    template = load_prompt_template()
    Q, omega, meta = query_llm_multiple(api_key, template, tickers, combined, n_queries, model, alpha, beta)
    return Q*21.0, omega*441.0, meta