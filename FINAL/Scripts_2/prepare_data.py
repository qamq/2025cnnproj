from multiprocessing import Process, Manager
import time
import gc
from Data import generate_chart as gc
from Data import equity_data as eqd

chart_type = "bar"
ws = 20
freq = "week"
ma_lags = [ws]
vb = True
country="CN"

def get_ret_by_year(year, semaphore):
    with semaphore:
        print(f"[{year}] Loading us_ret...")
        if country == "CN":
            ret = eqd.processed_CN_data()
        if country == "USA":
            ret = eqd.processed_US_data()
        mask = ret.index.get_level_values("Date").year.isin([year, year - 1, year - 2])
        df = ret[mask].copy()
        del ret
        print(f"[{year}] Finished loading and cleared us_ret")
        return df

def process_year(year, semaphore):
    print(f"Start processing year {year}")
    process_data = get_ret_by_year(year, semaphore)
    dgp_obj = gc.GenerateStockData(
        "CN", year, ws, freq,
        chart_freq=1, ma_lags=ma_lags, volume_bar=vb,
        need_adjust_price=True, allow_tqdm=False,
        chart_type=chart_type, process_data=process_data
    )
    dgp_obj.save_annual_data()
    dgp_obj.save_annual_ts_data()
    print(f"[{year}] Done.")

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

if __name__ == "__main__":
    year_list = list(range(2020, 2026))  # Replace with any list of years
    
    with Manager() as manager:
        semaphore = manager.Semaphore(1)
        for year_chunk in chunked(year_list, 6):
            processes = []
            for year in year_chunk:
                p = Process(target=process_year, args=(year, semaphore))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()



# (base) ronaldliu@Ronalds-MacBook-Pro Scripts_2 % python3 prepare_data.py
# Start processing year 2019
# [2019] Loading us_ret...
# Loading processed data from ../WORK_SPACE/data/processed_data/cn_ret.feather
# Finish loading in 0.02 min
# [2019] Finished loading and cleared us_ret
# Generating 20d_week_has_vb_[20]_ma_2019
# ['000001' '000002' '000004' ... '900955' '900956' '900957']
# [2019] [500/3880] Estimated time remaining: 511.78 seconds

# (base) ronaldliu@Ronalds-MacBook-Pro Scripts_2 % python3 prepare_data.py
# Start processing year 2019
# [2019] Loading us_ret...
# Loading processed data from ../WORK_SPACE/data/processed_data/us_ret.feather
# Finish loading processed data in 0.10 min
# [2019] Finished loading and cleared us_ret
# Generating 20d_week_has_vb_[20]_ma_2019
# ['10001' '10025' '10026' ... '93429' '93434' '93436']
# [2019] [500/9025] Estimated time remaining: 1322.83 seconds