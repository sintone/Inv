# -*- coding: utf-8 -*-
"""
@Description: 美股技术筛选 (GitHub Actions 适配版)
"""
import pandas as pd
import datetime
import numpy as np
import time
import talib
import requests
import json
import os
import traceback
from tqdm import tqdm
import akshare as ak

# ================= 配置区域 (改为从环境变量读取) =================
# 代理商鉴权信息 (请在 GitHub Secrets 中配置)
API_KEY = os.environ.get("PROXY_API_KEY", "L4U5680Y") 
API_PWD = os.environ.get("PROXY_API_PWD", "412E4561D849")

# 飞书 Webhook 地址
FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")

# 代理提取API
PROXY_API_URL = f"https://share.proxy.qg.net/get?key={API_KEY}&pwd={API_PWD}&num=1&area=&isp=0&format=json&distinct=true"
# 白名单添加API
WHITELIST_API_URL = "https://proxy.qg.net/whitelist/add"

# 最大重试次数
MAX_RETRY_PER_STOCK = 5

# 基础请求头
COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
# ===========================================

# ================= 日志与通知工具类 =================
class LogRecorder:
    """日志记录器：同时打印到控制台和保存到内存"""
    def __init__(self):
        self.logs = []

    def print(self, *args, **kwargs):
        """替代系统的 print 函数"""
        print(*args, **kwargs)
        # 保存到列表，用于发送通知
        msg = " ".join(map(str, args))
        self.logs.append(msg)

    def get_full_log(self):
        return "\n".join(self.logs)

    def send_feishu(self):
        """发送飞书通知"""
        content = self.get_full_log()
        if not content: return
        if not FEISHU_WEBHOOK_URL:
            print(">> [提示] 未配置 FEISHU_WEBHOOK，跳过发送。")
            return
            
        print("\n正在发送飞书通知...")
        headers = {"Content-Type": "application/json"}
        
        # 针对长消息进行简单截断
        if len(content) > 30000:
            content = content[:30000] + "\n...(消息过长已截断)"

        data = {
            "msg_type": "text",
            "content": {"text": content}
        }
        
        try:
            resp = requests.post(FEISHU_WEBHOOK_URL, headers=headers, json=data, timeout=10)
            if resp.status_code == 200:
                print(" -> 飞书通知发送成功！")
            else:
                print(f" -> 飞书通知发送失败: {resp.text}")
        except Exception as e:
            print(f" -> 发送过程出错: {e}")

# 初始化全局日志记录器
recorder = LogRecorder()
# ========================================================

class ProxyManager:
    """代理IP管理器"""
    def __init__(self, proxy_api_url, whitelist_api_url, api_key, api_pwd, max_retry=3):
        self.proxy_api_url = proxy_api_url
        self.whitelist_api_url = whitelist_api_url
        self.api_key = api_key
        self.api_pwd = api_pwd
        self.max_retry = max_retry

    def get_public_ip(self):
        """查询本机公网IP"""
        recorder.print(" -> 正在查询本机公网IP...")
        services = [
            {"url": "http://checkip.amazonaws.com", "type": "text"}, # 海外环境首选
            {"url": "https://api.ipify.org", "type": "text"},
            {"url": "http://members.3322.org/dyndns/getip", "type": "text"},
        ]

        for service in services:
            url = service["url"]
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    ip = resp.text.strip()
                    if ip and ip.count('.') == 3:
                        recorder.print(f"    [√] 获取成功: {ip}")
                        return ip
            except Exception:
                continue
        return None

    def auto_whitelist_current_ip(self):
        """核心逻辑：获取IP并加入白名单"""
        current_ip = self.get_public_ip()
        
        if not current_ip:
            recorder.print(" [错误] 无法获取本机公网IP，无法添加白名单。")
            return False
        
        recorder.print(f" -> 正在将 {current_ip} 添加至白名单...")
        params = {
            "Key": self.api_key,
            "Pwd": self.api_pwd,
            "IP": current_ip
        }
        
        try:
            resp = requests.get(self.whitelist_api_url, params=params, timeout=10)
            try:
                result = resp.json()
            except:
                recorder.print(f" [警告] 白名单接口返回非JSON: {resp.text}")
                return False

            code = result.get("Code")
            if code == 0:
                recorder.print(f" [成功] IP {current_ip} 已添加至白名单。")
                return True
            elif code == -202:
                recorder.print(f" [提示] IP {current_ip} 已存在或白名单已满 (Code: -202)。")
                return True
            else:
                recorder.print(f" [失败] 白名单添加失败，错误码: {code}, 描述: {result}")
                return False
        except Exception as e:
            recorder.print(f" [异常] 请求白名单接口出错: {e}")
            return False

    def get_valid_proxy(self):
        """获取有效的代理IP"""
        recorder.print("正在获取新的代理IP...")
        for attempt in range(self.max_retry):
            try:
                resp = requests.get(self.proxy_api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                try:
                    proxy_json = resp.json()
                except:
                    continue
                
                if isinstance(proxy_json, dict):
                    if str(proxy_json.get('Code', '')) == '-14':
                         recorder.print(f" -> [代理错误] 验证失败 (Code -14)。")
                         return None

                if "data" in proxy_json and len(proxy_json["data"]) > 0:
                    proxy_data = proxy_json["data"][0]
                    # 处理不同返回格式
                    if "server" in proxy_data:
                        server = proxy_data["server"]
                        ip, port = server.split(":")
                    elif "ip" in proxy_data and "port" in proxy_data:
                        ip = proxy_data["ip"]
                        port = proxy_data["port"]
                    else:
                        ip = proxy_data.get('ip')
                        port = proxy_data.get('port')

                    if not ip or not port: continue
                        
                    proxy = {
                        "http": f"http://{ip}:{port}", 
                        "https": f"http://{ip}:{port}"
                    }
                    return proxy
                else:
                    time.sleep(2)
            except:
                time.sleep(2)

        recorder.print(f" -> [警告] 获取代理失败，将尝试直连。")
        return None

def get_us_stock_daily_custom(symbol, start_date, end_date, proxy=None):
    """获取美股日线数据"""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    # 美股secid通常带有前缀，如 105.AAPL 或 106.MSFT
    # 如果传入的 symbol 已经带点（如 105.AAPL），直接使用
    # 如果不带，这里可能需要额外逻辑判断是 105 还是 106，但 akshare 获取的列表通常已经带了
    secid = symbol 
    
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "klt": "101", "fqt": "1", "secid": secid, "beg": start_date, "end": end_date,
    }
    try:
        r = requests.get(url, params=params, headers=COMMON_HEADERS, timeout=8, proxies=proxy)
        data_json = r.json()
        if not (data_json.get("data") and data_json["data"].get("klines")):
            return pd.DataFrame()
        
        temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["klines"]])
        temp_df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
                           "振幅", "涨跌幅", "涨跌额", "换手率"]
        temp_df["日期"] = pd.to_datetime(temp_df["日期"]).dt.strftime('%Y-%m-%d')
        numeric_cols = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
        temp_df[numeric_cols] = temp_df[numeric_cols].apply(pd.to_numeric)
        return temp_df
    except Exception as e:
        raise e

def print_simple_line(row):
    """
    极简格式输出 (修改为清晰的键值对格式)
    格式: 代码: Name (Code), 收盘价: ..., 涨跌幅: ...
    """
    line = (f"  代码: {row['name']} ({row['ts_code']}), "
            f"收盘价: {row['close']:.2f}, "
            f"涨跌幅: {row['pct_chg']}%, "
            f"7日涨跌幅: {row['day_7_chg']:.2f}%, "
            f"换手率: {row['turnover']}%")
    recorder.print(line)

def main():
    try:
        # 初始化
        proxy_manager = ProxyManager(PROXY_API_URL, WHITELIST_API_URL, API_KEY, API_PWD)
        
        # 步骤 -1
        recorder.print("步骤 -1: 初始化网络环境...")
        whitelist_success = proxy_manager.auto_whitelist_current_ip()
        
        if not whitelist_success:
            recorder.print("注意：白名单添加失败，后续可能会连接超时。")
        
        time.sleep(2) 
        current_proxy = proxy_manager.get_valid_proxy()
        
        # 步骤 0
        recorder.print("\n步骤 0 - 构建美股股票池...")
        all_pools = []
        
        # 1. 知名美股
        try:
            t = ak.stock_us_famous_spot_em(symbol='科技类') 
            if not t.empty:
                all_pools.append(t[['代码', '名称']].rename(columns={'代码': 'ts_code', '名称': 'name'}))
        except: pass
        
        # 2. 纳指100
        try:
            ndx = ak.index_stock_us_cons_em(symbol="NDX")
            if not ndx.empty:
                all_pools.append(ndx[['代码', '名称']].rename(columns={'代码': 'ts_code', '名称': 'name'}))
        except: pass

        # 3. 自定义列表
        CUSTOM_DICT = {
            'META': '105.META', 'RBLX': '106.RBLX', 'DIS': '106.DIS', 'NFLX': '105.NFLX',
            'SPOT': '106.SPOT', 'EA': '105.EA', 'GOOGL': '105.GOOGL', 'PINS': '106.PINS',
            'W': '106.W', 'TSLA': '105.TSLA', 'FL': '106.FL', 'ABNB': '105.ABNB',
            'GPRO': '105.GPRO', 'F': '106.F', 'JWN': '106.JWN', 'THO': '106.THO',
            'AMZN': '105.AMZN', 'BKNG': '105.BKNG', 'UAA': '106.UAA', 'ETSY': '105.ETSY',
            'EXPE': '105.EXPE', 'LEN': '106.LEN', 'SBUX': '105.SBUX', 'VFC': '106.VFC',
            'CMG': '106.CMG', 'SCI': '106.SCI', 'COLM': '105.COLM', 'BBY': '106.BBY',
            'WSM': '106.WSM', 'DHI': '106.DHI', 'GRMN': '106.GRMN', 'SKX': '106.SKX',
            'MAT': '105.MAT', 'MCD': '106.MCD', 'M': '106.M', 'NKE': '106.NKE',
            'HD': '106.HD', 'LULU': '105.LULU', 'IRBT': '105.IRBT', 'VSCO': '106.VSCO',
            'ADT': '106.ADT', 'GT': '105.GT', 'HAS': '105.HAS', 'COST': '105.COST',
            'WMT': '106.WMT', 'KO': '106.KO', 'KDP': '105.KDP', 'PEP': '105.PEP',
            'PG': '106.PG', 'CL': '106.CL', 'CVX': '106.CVX', 'WFC': '106.WFC',
            'BAC': '106.BAC', 'JPM': '106.JPM', 'GS': '106.GS', 'BLK': '106.BLK',
            'MSCI': '106.MSCI', 'C': '106.C', 'SPGI': '106.SPGI', 'MS': '106.MS',
            'CME': '105.CME', 'MORN': '105.MORN', 'CBOE': '107.CBOE', 'MCO': '106.MCO',
            'ICE': '106.ICE', 'IBKR': '105.IBKR', 'NDAQ': '105.NDAQ', 'ALGN': '105.ALGN',
            'VEEV': '106.VEEV', 'ILMN': '105.ILMN', 'MDT': '106.MDT', 'STAA': '105.STAA',
            'GILD': '105.GILD', 'COO': '105.COO', 'JNJ': '106.JNJ', 'IDXX': '105.IDXX',
            'DXCM': '105.DXCM', 'ABT': '106.ABT', 'PFE': '106.PFE', 'TMO': '106.TMO',
            'ISRG': '105.ISRG', 'GE': '106.GE', 'CAT': '106.CAT', 'RTX': '106.RTX',
            'LYFT': '105.LYFT', 'JCI': '106.JCI', 'LMT': '106.LMT', 'HON': '105.HON',
            'UPS': '106.UPS', 'FDX': '106.FDX', 'CAR': '105.CAR', 'ITW': '106.ITW',
            'WCN': '106.WCN', 'JBHT': '105.JBHT', 'RSG': '106.RSG', 'SWK': '106.SWK',
            'WM': '106.WM', 'TRI': '106.TRI', 'UBER': '106.UBER', 'GD': '106.GD',
            'MMM': '106.MMM', 'SHOP': '106.SHOP', 'U': '106.U', 'NVDA': '105.NVDA',
            'AFRM': '105.AFRM', 'TWLO': '106.TWLO', 'ON': '105.ON', 'SQ': '106.SQ',
            'MU': '105.MU', 'CRM': '106.CRM', 'QCOM': '105.QCOM', 'AMD': '105.AMD',
            'ZM': '105.ZM', 'LOGI': '105.LOGI', 'INTU': '105.INTU', 'AAPL': '105.AAPL',
            'ADBE': '105.ADBE', 'CSCO': '105.CSCO', 'AVGO': '105.AVGO', 'PYPL': '105.PYPL',
            'MSFT': '105.MSFT', 'V': '106.V', 'TSM': '106.TSM', 'ADSK': '105.ADSK',
            'LRCX': '105.LRCX', 'ORCL': '106.ORCL', 'DELL': '106.DELL', 'MA': '106.MA',
            'INTC': '105.INTC', 'WFG': '106.WFG', 'LAC': '106.LAC', 'IP': '106.IP',
            'CCK': '106.CCK', 'IFF': '106.IFF', 'SHW': '106.SHW', 'OPEN': '105.OPEN',
            'RDFN': '105.RDFN', 'ZG': '105.ZG', 'SQQQ': '105.SQQQ', 'YINN': '107.YINN',
            'PDD': '105.PDD', 'BABA': '106.BABA', 'BITX': '107.BITX', 'IBIT': '105.IBIT',
            'IWY': '107.IWY', 'MOAT': '107.MOAT', 'DRAG': '107.DRAG', 'CWEB': '107.CWEB',
            'GLD': '107.GLD'
        }
        c_df = pd.DataFrame([{'ts_code': v, 'name': k} for k, v in CUSTOM_DICT.items()])
        all_pools.append(c_df)
            
        stock_pool_df = pd.concat(all_pools, ignore_index=True)
        stock_pool_df.drop_duplicates(subset=['ts_code'], inplace=True)
        recorder.print(f"步骤 0 - 完成。共 {len(stock_pool_df)} 只股票。")

        # 步骤 1
        recorder.print("\n步骤 1 - 确定最新交易日...")
        today = datetime.datetime.now()
        end_date_str = today.strftime('%Y%m%d')
        start_check = (today - datetime.timedelta(days=15)).strftime('%Y%m%d')
        
        try:
            # 尝试获取苹果数据来确定交易日
            sample = get_us_stock_daily_custom('105.AAPL', start_check, end_date_str, proxy=current_proxy)
            
            # 如果失败，重试一次代理
            if sample.empty:
                current_proxy = proxy_manager.get_valid_proxy()
                sample = get_us_stock_daily_custom('105.AAPL', start_check, end_date_str, proxy=current_proxy)
            
            if sample.empty:
                raise Exception("无法获取样本数据")

            last_trade_day = pd.to_datetime(sample.iloc[-1]['日期'])
            last_trade_day_str = last_trade_day.strftime('%Y-%m-%d')
            last_trade_day_ak_str = last_trade_day.strftime('%Y%m%d')
            recorder.print(f"步骤 1 - 最新交易日: {last_trade_day_str}")
        except:
            recorder.print("错误: 无法获取基础行情，请检查网络/代理。")
            recorder.send_feishu() 
            return

        # 步骤 2
        recorder.print("\n步骤 2 - 计算均线及各项指标...")
        start_date_ma_str = (last_trade_day - pd.DateOffset(days=300)).strftime('%Y%m%d')
        results_list = []
        proxy_fail_count = 0

        pbar = tqdm(stock_pool_df.iterrows(), total=stock_pool_df.shape[0], desc="Processing")
        
        for _, row in pbar:
            stock_code = row['ts_code']
            stock_name = row['name']
            
            hist_df = pd.DataFrame()
            fetched = False
            
            for retry in range(MAX_RETRY_PER_STOCK):
                try:
                    hist_df = get_us_stock_daily_custom(stock_code, start_date_ma_str, last_trade_day_ak_str, proxy=current_proxy)
                    fetched = True
                    proxy_fail_count = 0
                    break
                except:
                    proxy_fail_count += 1
                    if proxy_fail_count >= 3:
                        current_proxy = proxy_manager.get_valid_proxy()
                        proxy_fail_count = 0
            
            if not fetched or hist_df.empty or len(hist_df) < 125: continue
            if hist_df.iloc[-1]['日期'] != last_trade_day_str: continue

            try:
                closes = hist_df['收盘'].values.astype(float)
                
                ma20 = talib.SMA(closes, 20)
                ma60 = talib.SMA(closes, 60)
                ma120 = talib.SMA(closes, 120)
                
                if np.isnan([ma20[-1], ma60[-1], ma120[-1]]).any(): continue

                c_now = closes[-1]
                m20_now, m60_now, m120_now = ma20[-1], ma60[-1], ma120[-1]
                c_prev = closes[-2]
                m20_prev, m60_prev, m120_prev = ma20[-2], ma60[-2], ma120[-2]
                
                max_ma_now = max(m20_now, m60_now, m120_now)
                min_ma_now = min(m20_now, m60_now, m120_now)
                max_ma_prev = max(m20_prev, m60_prev, m120_prev)
                min_ma_prev = min(m20_prev, m60_prev, m120_prev)

                is_cross_up = (c_now > max_ma_now) and (c_prev <= max_ma_prev)
                is_cross_down = (c_now < min_ma_now) and (c_prev >= min_ma_prev)
                is_bull_sustain = (c_now > max_ma_now) and (c_prev > max_ma_prev)
                is_bear_sustain = (c_now < min_ma_now) and (c_prev < min_ma_prev)
                
                bias_up_val = (c_now - max_ma_now) / max_ma_now
                is_bias_up = bias_up_val > 0.15 
                
                bias_down_val = (min_ma_now - c_now) / min_ma_now
                is_bias_down = bias_down_val > 0.15

                day_7_chg = 0.0
                if len(hist_df) >= 8:
                    day_7_chg = (c_now - closes[-8]) / closes[-8] * 100

                res = {
                    'ts_code': stock_code, 'name': stock_name, 
                    'close': c_now, 'pct_chg': hist_df.iloc[-1]['涨跌幅'], 'turnover': hist_df.iloc[-1]['换手率'],
                    'day_7_chg': day_7_chg,
                    'flag_cross_up': is_cross_up,
                    'flag_cross_down': is_cross_down,
                    'flag_bull_sustain': is_bull_sustain,
                    'flag_bear_sustain': is_bear_sustain,
                    'flag_bias_up_15': is_bias_up,
                    'flag_bias_down_15': is_bias_down,
                }
                results_list.append(res)
            except: pass
                
        pbar.close()
        
        if not results_list:
            recorder.print("没有数据。")
            recorder.send_feishu()
            return

        master_df = pd.DataFrame(results_list)

        # 步骤 3
        recorder.print("\n" + "="*60)
        recorder.print("筛选结果 (包含7日涨跌幅)")
        recorder.print("="*60)
        
        categories = [
            ("收盘价上穿三条均线", master_df['flag_cross_up']),
            ("收盘价下穿三条均线", master_df['flag_cross_down']),
            ("上涨乖离率 > 15%", master_df['flag_bias_up_15']),
            ("下跌乖离率 > 15%", master_df['flag_bias_down_15']),
            ("持续多头形态", master_df['flag_bull_sustain']),
            ("持续空头形态", master_df['flag_bear_sustain']),
        ]

        for title, mask in categories:
            df_sub = master_df[mask]
            recorder.print(f"\n=== {title} (共 {len(df_sub)} 只) ===")
            if df_sub.empty:
                recorder.print("  (无符合条件股票)")
            else:
                for _, row in df_sub.sort_values('day_7_chg', ascending=False).iterrows():
                    print_simple_line(row)

        top10_gainers = master_df.sort_values('day_7_chg', ascending=False).head(10)
        top10_losers = master_df.sort_values('day_7_chg', ascending=True).head(10)

        recorder.print(f"\n=== 7日涨幅榜前10 ===")
        for _, row in top10_gainers.iterrows():
            print_simple_line(row)

        recorder.print(f"\n=== 7日跌幅榜前10 ===")
        for _, row in top10_losers.iterrows():
            print_simple_line(row)
            
        output_file = f'US_Simple_Screener_{last_trade_day_ak_str}.csv'
        master_df.to_csv(output_file, index=False, encoding='utf-8-sig', float_format='%.2f')
        recorder.print(f"\n结果已保存: {output_file}")
        
        # === 最后一步：发送所有积累的日志 ===
        recorder.send_feishu()

    except KeyboardInterrupt:
        recorder.print("\n程序中断")
        recorder.send_feishu()
    except Exception as e:
        import traceback
        traceback.print_exc()
        recorder.print(f"程序发生严重错误: {e}")
        recorder.send_feishu()

if __name__ == '__main__':
    main()
