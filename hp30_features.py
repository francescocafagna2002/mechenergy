"""HP30: household heat-response features (kW profiles; daily energies in kWh).

Definition table: name | formula | HP direction | reason | main confounder.
E=daily import; T=mean temperature; H=max(0,tau-T); B=median warm E.
W=Nov-Feb; S=May-Aug; heating season=Oct-Apr; night=00-06.
All ratios require positive denominators. See compute_features for exact guards.

hp_daily_import_temperature_corr | corr(E,T), Oct-Apr | negative | heating | resistance/PV
hp_hdd_year_slope_stability | 1/(1+CV annual slopes) | higher | repeated response | resistance
hp_balance_temperature_c | best interior tau, grid 8:0.5:20 | plausible | thermal balance | PV/occupancy
hp_hdd_slope_kwh_per_degree_day | OLS slope E~H, nonnegative | higher | heating magnitude | resistance
hp_hdd_slope_normalized | slope/B | higher | relative heating | resistance
hp_hdd_response_r2 | 1-SSE/SST | higher | coherent response | resistance/PV
hp_cold_mild_daily_energy_ratio | mean E cold / mean E warm | higher | temperature effect | PV
hp_winter_summer_daily_energy_ratio | mean E W / mean E S | higher | seasonality | PV/EV
hp_heating_excess_share_winter | sum max(0,E-B) W / sum E W | higher | winter excess | PV
hp_cold_day_heating_excess_kwh | median max(0,E-B) cold | higher | heating size | resistance
hp_cold_day_active_fraction | fraction cold excess>A | higher | persistence | resistance
hp_winter_baseload_increase_kw | P20 W - P20 S | higher | sustained load | PV
hp_winter_load_factor | mean W/P95 W | higher | sustained load | commercial
hp_winter_minus_summer_load_factor | LF W-LF S | higher | seasonal persistence | commercial
hp_cold_night_temperature_corr | corr(night E,T), Oct-Apr | negative | night heating | resistance
hp_night_day_temperature_sensitivity_ratio | slope(night E/6)/slope(day E/18) | near 1 | all-day heating | resistance
hp_cold_weather_high_load_hours_per_day | cold hours above warm slot baseline+threshold | higher | duration | resistance
hp_heating_season_persistence | fraction eligible weeks >=half cold days active | higher | persistence | resistance

general_mean_daily_import_kwh | mean E | size | household scale | commercial
general_median_import_power_kw | median P | size | baseload | commercial
general_p95_import_power_kw | P95 P | size | peak scale | EV
general_daily_energy_cv | std(E)/mean(E) | contextual | variability | occupancy
general_night_day_energy_ratio | sum night E/sum day E | contextual | timing | boiler
general_weekend_weekday_energy_ratio | mean weekend E/mean weekday E | contextual | occupancy | commercial
general_load_factor_all_year | mean P/P95 P | contextual | sustained load | commercial

conf_export_to_import_ratio | sum export/sum import, common days | contextual | export | PV
conf_export_irradiance_corr | summer corr(export,irradiation) | contextual | solar | PV
conf_night_export_share | export 20-06/total export | contextual | dark export | storage
conf_evening_long_plateau_score | hour share*4p(1-p)*(1-max(0,corr(hours,H))^2) | EV flag | intermittent plateau | scheduled heat
conf_fixed_time_daily_load_concentration | normalized max 2h event-start share*active-day share | boiler flag | scheduled load | scheduled HP

HDD correlation was replaced: with one predictor and intercept, its square is
R-squared. Annual stability adds distinct information, but needs two full years.
No compressor-cycle claim is made from aggregate 15-minute measurements.
Labels are NEVER read here. GIGI location columns only are read by PV25 helpers.
"""
from __future__ import annotations
import argparse
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'PV'))
import pv20_features as p
import pv25_features as w

HP_FEATURES = [
 'hp_daily_import_temperature_corr','hp_hdd_year_slope_stability',
 'hp_balance_temperature_c','hp_hdd_slope_kwh_per_degree_day',
 'hp_hdd_slope_normalized','hp_hdd_response_r2','hp_cold_mild_daily_energy_ratio',
 'hp_winter_summer_daily_energy_ratio','hp_heating_excess_share_winter',
 'hp_cold_day_heating_excess_kwh','hp_cold_day_active_fraction',
 'hp_winter_baseload_increase_kw','hp_winter_load_factor',
 'hp_winter_minus_summer_load_factor','hp_cold_night_temperature_corr',
 'hp_night_day_temperature_sensitivity_ratio','hp_cold_weather_high_load_hours_per_day',
 'hp_heating_season_persistence']
GENERAL_FEATURES = ['general_mean_daily_import_kwh','general_median_import_power_kw',
 'general_p95_import_power_kw','general_daily_energy_cv','general_night_day_energy_ratio',
 'general_weekend_weekday_energy_ratio','general_load_factor_all_year']
CONF_FEATURES = ['conf_export_to_import_ratio','conf_export_irradiance_corr',
 'conf_night_export_share','conf_evening_long_plateau_score',
 'conf_fixed_time_daily_load_concentration']
MODEL_FEATURES = HP_FEATURES + GENERAL_FEATURES + CONF_FEATURES
ELECTRICAL_HP_FEATURES = [HP_FEATURES[i] for i in (7,11,12,13)]
MIN_DAYS = 20
NIGHT = np.arange(96) < 24
EXPORT_NIGHT = (np.arange(96) < 24) | (np.arange(96) >= 80)
EV_NIGHT = (np.arange(96) < 24) | (np.arange(96) >= 72)


def ratio(a, b, floor=1e-9):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > floor else np.nan


def corr(a, b, minimum=20, min_span=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < minimum or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    if np.ptp(b) < min_span:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def linear_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    var = np.sum((x-x.mean())**2)
    if var <= 1e-10:
        return np.nan, np.nan, np.nan
    beta = max(0., float(np.sum((x-x.mean())*(y-y.mean())) / var))
    intercept = float(y.mean() - beta*x.mean())
    sse = float(np.sum((y - intercept - beta*x)**2))
    return beta, intercept, sse


def fit_hdd(energy, temperature):
    e, t = np.asarray(energy, float), np.asarray(temperature, float)
    good = np.isfinite(e) & np.isfinite(t)
    e, t = e[good], t[good]
    if len(e) < 60 or np.percentile(t,90)-np.percentile(t,10) < 8:
        return None
    fits = []
    for tau in np.arange(8., 20.01, .5):
        if np.sum(t <= tau-2) < 20 or np.sum(t >= tau) < 20:
            continue
        beta, intercept, sse = linear_fit(np.maximum(0,tau-t),e)
        fits.append((sse, float(tau), beta, intercept))
    if not fits:
        return None
    sse, tau, beta, intercept = min(fits)
    sst = float(np.sum((e-e.mean())**2))
    r2 = max(0., 1-sse/sst) if sst > 1e-9 else np.nan
    return dict(tau=tau, beta=beta, intercept=intercept, r2=r2,
                identified=bool(np.isfinite(r2) and r2 >= .1 and 8 < tau < 20))


def normal_days(index):
    """The raw 96-column layout cannot represent DST 92/100 slots reliably.
    Exclude those electrical days; weather itself retains true 23/25 hours.
    """
    idx = pd.DatetimeIndex(index)
    midnight = idx.tz_localize('Europe/Zurich')
    tomorrow = (idx+pd.Timedelta(days=1)).tz_localize('Europe/Zurich')
    return np.asarray((tomorrow-midnight).total_seconds() == 86400)


def complete_profiles(profiles):
    q = profiles.sort_index().astype(float).copy()
    valid = np.isfinite(q.to_numpy()).all(axis=1) & (q.to_numpy() >= 0).all(axis=1)
    return q.loc[valid & normal_days(q.index)]


def daily_table(profiles):
    q = complete_profiles(profiles)
    return pd.DataFrame({
        'daily_import_kwh': q.sum(axis=1)/4,
        'nighttime_import_kwh': q.loc[:,NIGHT].sum(axis=1)/4,
        'daytime_import_kwh': q.loc[:,~NIGHT].sum(axis=1)/4,
        'daily_min_kw': q.min(axis=1), 'daily_mean_kw': q.mean(axis=1),
        'daily_p95_kw': q.quantile(.95,axis=1)},index=q.index)


def events(profiles, threshold):
    """Contiguous residual-power episodes; never bridge missing days or slots."""
    if profiles.empty:
        return []
    baseline = np.quantile(profiles.to_numpy(),.2)
    result = []
    cuts = np.r_[0, np.flatnonzero(np.diff(profiles.index.values).astype('timedelta64[D]').astype(int) != 1)+1, len(profiles)]
    for lo, hi in zip(cuts[:-1],cuts[1:]):
        a = profiles.iloc[lo:hi].to_numpy().ravel()
        mask = a >= baseline + threshold
        edges = np.diff(np.r_[False, mask, False].astype(int))
        for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            # Truncated boundary events cannot establish a start or duration.
            if start == 0 or end == len(a):
                continue
            result.append((lo, start, end, a[start:end]))
    return result


def compute_features(import_profiles, export_profiles=None, weather=None):
    """Pure feature computation, with no location/label/file dependencies.
    Requires 96 finite nonnegative slots/day, >=20 days per simple estimate,
    >=60 paired days and >=8C interdecile span for HDD. No zero imputation.
    """
    q = complete_profiles(import_profiles)
    f = dict.fromkeys(MODEL_FEATURES, np.nan)
    diag = {'n_days':len(q),'n_common_weather_days':0,'n_cold_days':0,'n_warm_days':0,
            'n_dst_excluded':int((~normal_days(import_profiles.index)).sum())}
    span = (import_profiles.index.max()-import_profiles.index.min()).days+1 if len(import_profiles) else 0
    diag['data_coverage_ratio'] = ratio(len(q),span)
    if len(q) < 20:
        diag['feature_error'] = 'fewer than 20 complete normal electrical days'
        return f, diag, daily_table(import_profiles)
    d = daily_table(q)
    if weather is None:
        weather = pd.DataFrame(columns=['temperature_c','irradiation_kwh_m2'],index=pd.DatetimeIndex([]))
    d = d.join(weather[['temperature_c','irradiation_kwh_m2']],how='left')
    e = d.daily_import_kwh
    n = d.nighttime_import_kwh
    day = d.daytime_import_kwh
    t = d.temperature_c
    winter = d.index.month.isin([11,12,1,2])
    summer = d.index.month.isin([5,6,7,8])
    heat = d.index.month.isin([10,11,12,1,2,3,4])
    weekend = d.index.dayofweek >= 5
    power = q.to_numpy()
    lf = lambda a: ratio(np.mean(a),np.quantile(a,.95)) if a.size else np.nan
    for key,value in zip(GENERAL_FEATURES,[e.mean(),np.median(power),np.quantile(power,.95),
            ratio(e.std(ddof=0),e.mean()),ratio(n.sum(),day.sum()),
            ratio(e[weekend].mean(),e[~weekend].mean()) if min(sum(weekend),sum(~weekend))>=20 else np.nan,lf(power)]):
        f[key] = value
    if sum(winter)>=20:
        f[HP_FEATURES[12]] = lf(power[winter])
    if min(sum(winter),sum(summer))>=20:
        f[HP_FEATURES[7]] = ratio(e[winter].mean(),e[summer].mean())
        f[HP_FEATURES[11]] = np.quantile(power[winter],.2)-np.quantile(power[summer],.2)
        f[HP_FEATURES[13]] = lf(power[winter])-lf(power[summer])
    f[HP_FEATURES[0]] = corr(e[heat],t[heat],min_span=3)
    f[HP_FEATURES[14]] = corr(n[heat],t[heat],min_span=3)
    diag['n_common_weather_days'] = int(t.notna().sum())
    fit = fit_hdd(e,t)
    tau = fit['tau'] if fit else np.nan
    trusted_tau = tau if fit and fit['identified'] else 18.
    warm = t >= max(18.,trusted_tau)
    cold = t <= min(8.,trusted_tau-4)
    diag.update(n_cold_days=int(cold.sum()),n_warm_days=int(warm.sum()),
                balance_candidate_c=tau,balance_identified=bool(fit and fit['identified']))
    b = float(e[warm].median()) if warm.sum()>=20 else np.nan
    diag['warm_day_baseline_kwh'] = b
    excess = (e-b).clip(lower=0)
    mad = float(np.median(np.abs(e[warm]-b))) if np.isfinite(b) else np.nan
    active_threshold = max(3.,.15*b,3*1.4826*mad) if np.isfinite(b) else np.nan
    diag['active_excess_threshold_kwh'] = active_threshold
    if fit:
        f[HP_FEATURES[2]] = tau if fit['identified'] else np.nan
        f[HP_FEATURES[3]] = fit['beta']
        f[HP_FEATURES[4]] = ratio(fit['beta'],b,floor=1.)
        f[HP_FEATURES[5]] = fit['r2']
        good = t.notna()
        hdd = np.maximum(0,tau-t[good])
        bn = linear_fit(hdd,n[good]/6)[0]
        bd = linear_fit(hdd,day[good]/18)[0]
        f[HP_FEATURES[15]] = ratio(bn,bd,floor=.01)
        slopes = []
        if fit['identified']:
            for year in sorted(d.index.year.unique()):
                k = good & (d.index.year==year)
                tk = t[k]
                if k.sum()>=60 and (tk<=tau-2).sum()>=20 and (tk>=tau).sum()>=20:
                    slopes.append(linear_fit(np.maximum(0,tau-tk),e[k])[0])
        if len(slopes)>=2 and np.mean(slopes)>1e-6:
            f[HP_FEATURES[1]] = 1/(1+np.std(slopes)/np.mean(slopes))
    if np.isfinite(b):
        if sum(winter)>=20:
            f[HP_FEATURES[8]] = ratio(excess[winter].sum(),e[winter].sum())
        if cold.sum()>=20:
            f[HP_FEATURES[6]] = ratio(e[cold].mean(),e[warm].mean())
            f[HP_FEATURES[9]] = excess[cold].median()
            f[HP_FEATURES[10]] = float((excess[cold]>active_threshold).mean())
            warm_power = power[warm]
            slot_base = np.median(warm_power,axis=0)
            slot_mad = np.median(np.abs(warm_power-slot_base),axis=0)
            high = slot_base + np.maximum(.3,3*1.4826*slot_mad)
            f[HP_FEATURES[16]] = float(np.mean(np.sum(power[cold]>high,axis=1)/4))
            z = pd.DataFrame({'cold':cold & heat,'active':(excess>active_threshold) & cold & heat},index=d.index)
            weekly = z.resample('W-SUN').sum()
            eligible = weekly['cold']>=3
            if eligible.sum()>=4:
                f[HP_FEATURES[17]] = float((weekly.loc[eligible,'active']/weekly.loc[eligible,'cold']>=.5).mean())
    if export_profiles is not None:
        x = complete_profiles(export_profiles)
        common = q.index.intersection(x.index)
        if len(common)>=20:
            ex = x.loc[common]
            xe = ex.sum(axis=1)/4
            f[CONF_FEATURES[0]] = ratio(xe.sum(),e.loc[common].sum())
            f[CONF_FEATURES[2]] = ratio(ex.loc[:,EXPORT_NIGHT].to_numpy().sum()/4,xe.sum(),floor=.1)
            ss = common.month.isin([5,6,7,8])
            f[CONF_FEATURES[1]] = corr(xe.loc[ss],d.loc[common[ss],'irradiation_kwh_m2'])
    # EV confounder: flat, 2-16h, >=3 kW residual, >=75% in 18-06.
    hours = np.zeros(len(q))
    for lo,start,end,a in events(q,3.):
        slots = np.arange(start,end)
        if 8<=len(a)<=64 and np.std(a)/np.mean(a)<=.15 and EV_NIGHT[slots%96].mean()>=.75:
            np.add.at(hours,lo+slots//96,.25)
    active = np.mean(hours>0)
    if t.notna().sum()>=40 and np.ptp(t.dropna())>=8:
        h = np.maximum(0,(tau if fit else 15.)-t)
        r = corr(hours,h,minimum=40)
        # Zero event series is observed absence, not missing evidence.
        if not np.any(hours):
            f[CONF_FEATURES[3]] = 0.
        elif np.isfinite(r):
            f[CONF_FEATURES[3]] = float(hours.sum()/(24*len(q))*4*active*(1-active)*(1-max(0,r)**2))
    starts, active_dates = [], set()
    for lo,start,end,a in events(q,1.5):
        if len(a)>=2:
            starts.append(start%96)
            active_dates.add(lo+start//96)
    if len(starts)>=20:
        hist = np.bincount(starts,minlength=96)
        mass = max(np.r_[hist,hist][i:i+8].sum() for i in range(96))/len(starts)
        f[CONF_FEATURES[4]] = max(0.,(mass-1/12)/(1-1/12))*len(active_dates)/len(q)
    for key,value in f.items():
        f[key] = float(value) if np.isfinite(value) else np.nan
    return f,diag,d


class TemperatureRepository(w.WeatherRepository):
    """Reuse PV25 geocoding, STAC, downloads, cache, time alignment.
    Select stations by TEMPERATURE coverage on this house's valid dates.
    """
    def _station_daily_data(self, station):
        if station in self._station_errors:
            raise RuntimeError(self._station_errors[station])
        if station not in self._station_daily:
            try:
                # Filter irrelevant historical decades before PV25 download:
                # compatible even with older PV25 year-regex implementations.
                items = []
                for item in self.stac_items:
                    cp = dict(item)
                    assets = {}
                    for key,asset in item.get('assets',{}).items():
                        match = re.search(r'historical_([0-9]{4})-([0-9]{4})',asset.get('href',''))
                        if match and self.years and not any(int(match[1])<=y<=int(match[2]) for y in self.years):
                            continue
                        assets[key] = asset
                    cp['assets'] = assets
                    items.append(cp)
                hourly = w.download_station_hourly(station,self.cache_dir,items,refresh=self.refresh,years=self.years)
                if not w._find_column(hourly.columns,['gre000h0']):
                    hourly = hourly.assign(gre000h0=np.nan)
                self._station_daily[station] = w.weather_to_daily(hourly)
            except Exception as exc:
                self._station_errors[station] = str(exc)
                raise
        return self._station_daily[station]

    def for_house(self,plz,dates):
        ort = self.plz_to_ort.get(plz,'')
        canton = self.plz_to_canton.get(plz,'')
        lat,lon = w.geocode_plz(plz,ort,self.cache_dir,refresh=self.refresh,canton=canton)
        errors = []
        for _,row in w.rank_stations(self.stations,lat,lon).iterrows():
            station = str(row['station']).strip()
            try:
                daily = self._station_daily_data(station)
                if daily.temperature_c.reindex(dates).notna().sum()<60:
                    errors.append(station+': fewer than 60 matched temperature days')
                    continue
                return daily,dict(ort=ort,kanton=canton,weather_station=station,
                                  weather_station_distance_km=float(row['distance_km']))
            except Exception as exc:
                errors.append(station+': '+str(exc))
        raise RuntimeError('No usable temperature station: '+' | '.join(errors[:5]))


def _strict_profiles_from_parts(parts):
    """Combine already-parsed AEW meter rows using the validated HP30 rules.

    `parts` contains DataFrames with:
        mp_id, channel, date, 0..95

    A quarter-hour is valid after multi-meter aggregation only when every
    constituent meter row has a valid value for that quarter-hour.
    """
    if not parts:
        raise ValueError('No selected import/export rows found')

    all_rows = pd.concat(parts, ignore_index=True)
    slots = list(range(96))
    group = all_rows.groupby(['mp_id', 'channel', 'date'], sort=True)

    counts = group[slots].count()
    sums = group[slots].sum(min_count=1)

    # Same strict semantics as the original HP30 CSV implementation:
    # if one constituent meter is missing for a slot, aggregated slot = missing.
    return sums.where(counts.eq(group.size(), axis=0))


def _profile_frame(lab, values):
    """Convert one parsed chunk into the table expected by HP30."""
    values = np.array(values, dtype='float32', copy=True)

    # pv20._post() records raw negatives in n_neg and floors the counter.
    # The validated HP30 path then invalidates the complete raw meter-day row.
    if 'n_neg' in lab.columns:
        values[lab['n_neg'].to_numpy() > 0, :] = np.nan

    z = pd.DataFrame(values, columns=list(range(96)))
    for name in ['mp_id', 'channel', 'date']:
        z[name] = lab[name].to_numpy()
    return z


def load_profiles_csv(root, files, ids, cache_dir, workers=4):
    """Slow fallback: read raw CSVs through the NEW pv20 parser.

    pv20_features.py now handles the heterogeneous 2023 / 2024+ AEW layouts
    itself, so hp30 no longer uses pv25.raw_csv_layout().
    """
    ids = list(dict.fromkeys(str(x).strip() for x in ids if str(x).strip()))

    def one(path):
        chunks = []
        for lab, values in p.iter_aew_csv(
            str(path),
            mp_ids=ids,
            cache_dir=str(cache_dir) if cache_dir else None,
        ):
            chunks.append(_profile_frame(lab, values))
        return chunks

    parts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, chunks in enumerate(pool.map(one, files), 1):
            parts.extend(chunks)
            print(f'raw files {i}/{len(files)}', flush=True)

    profiles = _strict_profiles_from_parts(parts)

    # Location only; technology/label columns are never read.
    locations = w.mpid_plz_from_csv(
        root,
        ids,
        pattern=p.FILE_GLOB,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return profiles, locations


def load_profiles_from_store(store, ids):
    """FAST path: read selected MP IDs directly from the binary AEW store.

    This deliberately reuses pv20.read_store(), i.e. the same binary access
    layer used by the binary-ready PV pipeline.

    Important:
      * store v0..v95 are ALREADY kW -> no x4 conversion here;
      * only buckets containing the requested MP IDs are touched;
      * pv20._post() still applies the same counter/export preprocessing;
      * HP30 keeps its strict negative-row and multi-meter missing-slot rules.
    """
    required = ('read_store', 'first_store_ids', 'STORE_DEFAULT')
    missing = [name for name in required if not hasattr(p, name)]
    if missing:
        raise RuntimeError(
            'Your PV/pv20_features.py is not the binary-ready version. '
            'Missing: ' + ', '.join(missing)
        )

    ids = list(dict.fromkeys(str(x).strip() for x in ids if str(x).strip()))
    parts = []

    n_chunks = 0
    n_meter_days = 0

    for lab, values in p.read_store(str(store), mp_ids=ids):
        n_chunks += 1
        n_meter_days += len(lab)
        parts.append(_profile_frame(lab, values))

    profiles = _strict_profiles_from_parts(parts)

    # Same MP-ID -> PLZ bridge already used by binary-ready pv25_features.py.
    locations = w.mpid_plz_from_store(str(store), ids)

    print(
        f'store: {n_meter_days:,} raw meter-days for {len(ids)} houses '
        f'({n_chunks} bucket chunks)',
        flush=True,
    )

    return profiles, locations


def fill_missing_features(table):
    """Fill NaNs in the house x feature table the same way as the PV scripts.

    1. pv20.fill_missing() if available, so HP30 follows the shared convention.
       It may only know pv20's own columns, so we don't trust it blindly.
    2. Anything still missing -> median of that feature over all houses.
    3. A column with no value for any house cannot be filled -> dropped.
    """
    table = table.astype(float)
    if hasattr(p, 'fill_missing'):
        try:
            out = p.fill_missing(table.copy())
            # only accept it if it kept our rows/columns
            if isinstance(out, pd.DataFrame) and out.index.equals(table.index) \
                    and set(table.columns) <= set(out.columns):
                table = out[table.columns].astype(float)
        except Exception as exc:
            print(f'note: pv20.fill_missing not usable here ({exc}); using medians', flush=True)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.fillna(table.median())
    return table.dropna(axis=1, how='all')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # Same binary-first convention as the supplied binary-ready PV25 script.
    parser.add_argument(
        'root',
        nargs='?',
        help=(
            'raw AEW CSV root; not required when the binary store is used'
        ),
    )
    parser.add_argument(
        '--store',
        default=None,
        help=(
            "binary store built by build_store.py; default: ./aew_store "
            "when it exists; --store '' forces raw CSV mode"
        ),
    )
    parser.add_argument('--gigi')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--max-houses', type=int)
    sel.add_argument('--mp-ids')
    sel.add_argument('--mp-ids-file', type=Path)

    parser.add_argument('--max-files', type=int, help='CSV mode only')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache-dir', type=Path, default=HERE/'PV/aew_cache')
    parser.add_argument('--weather-cache', type=Path, default=HERE/'PV/.pv25_weather_cache')
    parser.add_argument('--refresh-weather', action='store_true')
    parser.add_argument('--daily-output-dir', type=Path)
    parser.add_argument('--keep-nan', action='store_true',
                        help='do not fill missing feature values (same flag as pv20)')
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=HERE.parent/'store/hp30_quick.csv',
    )

    args = parser.parse_args(argv)

    if args.workers < 1 or (args.max_houses is not None and args.max_houses < 1):
        parser.error('workers/max-houses must be positive')

    start = time.perf_counter()

    # ------------------------------------------------------------------
    # PICK INPUT BACKEND
    # ------------------------------------------------------------------
    default_store = getattr(p, 'STORE_DEFAULT', 'aew_store')

    if args.store == '':
        store = None
    elif args.store:
        candidate = Path(args.store)
        if not candidate.is_dir():
            parser.error(f"store not found: {candidate}")
        store = candidate
    else:
        candidate = Path(default_store)
        store = candidate if candidate.is_dir() else None

    default_root = getattr(w, 'DEFAULT_CSV_ROOT', '../aew-data/test-blob/input_data')
    root = args.root or (default_root if Path(default_root).is_dir() else None)

    default_gigi = getattr(
        w,
        'DEFAULT_GIGI',
        '../aew-data/test-blob/input_data/HackDays2026 - GIGI.csv',
    )
    gigi = args.gigi or (default_gigi if Path(default_gigi).is_file() else None)

    if store is None and not root:
        parser.error('Give a CSV root folder or an existing --store folder.')

    if store is not None:
        print(f'reading electricity from binary store: {store}', flush=True)
        if args.max_files is not None:
            print('note: --max-files is ignored in binary-store mode', flush=True)
    else:
        print(f'reading electricity from raw CSVs: {root}', flush=True)

    # ------------------------------------------------------------------
    # SELECT MP IDs
    # ------------------------------------------------------------------
    if args.mp_ids:
        ids = [x.strip() for x in args.mp_ids.split(',') if x.strip()]
    elif args.mp_ids_file:
        ids = []
        for line in args.mp_ids_file.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s or s.lower().startswith('mp'):
                continue
            ids.append(s.split(';')[0].split(',')[0].strip())
    elif store is not None:
        if not hasattr(p, 'first_store_ids'):
            parser.error(
                'PV/pv20_features.py is not binary-ready: first_store_ids() missing'
            )
        ids = p.first_store_ids(str(store), args.max_houses or 3)
    else:
        files_for_ids = p.find_files(root, max_files=args.max_files)
        ids = p.first_mp_ids(files_for_ids[0], args.max_houses or 3)

    ids = list(dict.fromkeys(str(x).strip() for x in ids if str(x).strip()))
    if not ids:
        parser.error('Empty MP selection')

    # ------------------------------------------------------------------
    # LOAD THE SAME 96-SLOT PROFILES FROM STORE OR CSV
    # ------------------------------------------------------------------
    if store is not None:
        profiles, locations = load_profiles_from_store(store, ids)
        source_description = f'binary store {store}'
    else:
        files = p.find_files(root, max_files=args.max_files)
        profiles, locations = load_profiles_csv(
            root, files, ids, args.cache_dir, args.workers
        )
        source_description = f'raw CSV {root}'

    print(
        f'Selected {len(ids)} MP IDs; labels not used; source={source_description}',
        flush=True,
    )

    all_dates = profiles.index.get_level_values('date').unique()

    # ------------------------------------------------------------------
    # WEATHER — unchanged
    # ------------------------------------------------------------------
    repo, error = None, ''
    try:
        repo = TemperatureRepository(
            str(args.weather_cache),
            gigi_path=gigi,
            refresh=args.refresh_weather,
            dates=all_dates,
        )
    except Exception as exc:
        error = str(exc)

    # ------------------------------------------------------------------
    # HP30 SCIENTIFIC FEATURES — unchanged
    # ------------------------------------------------------------------
    rows = []

    for mp in ids:
        empty = pd.DataFrame(
            columns=range(96),
            index=pd.DatetimeIndex([]),
        )

        def channel(name):
            try:
                return profiles.xs(
                    (mp, name),
                    level=('mp_id', 'channel'),
                )
            except KeyError:
                return empty.copy()

        imp, exp = channel('import'), channel('export')
        plz = locations.get(mp, '')

        info = dict(
            ort='',
            kanton='',
            weather_station='',
            weather_station_distance_km=np.nan,
        )
        weather, weather_error = None, error

        if repo is not None:
            try:
                valid_import = complete_profiles(imp)

                if len(valid_import) < 60:
                    raise ValueError(
                        'Fewer than 60 valid electrical days for station selection'
                    )
                if not plz:
                    raise ValueError('Missing or conflicting PLZ')

                weather, info = repo.for_house(
                    plz,
                    valid_import.index,
                )
            except Exception as exc:
                weather_error = str(exc)

        features, diag, daily = compute_features(imp, exp, weather)

        row = dict(
            mp_id=mp,
            plz=plz,
            **info,
            **diag,
            weather_error=weather_error,
            **features,
        )
        rows.append(row)

        if args.daily_output_dir:
            args.daily_output_dir.mkdir(parents=True, exist_ok=True)
            daily.to_csv(
                args.daily_output_dir/f'{mp}_daily.csv',
                index_label='date',
            )

        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in [
                        'mp_id',
                        'plz',
                        'ort',
                        'weather_station',
                        'weather_station_distance_km',
                        'n_days',
                        'n_common_weather_days',
                        HP_FEATURES[2],
                        HP_FEATURES[3],
                        HP_FEATURES[5],
                        HP_FEATURES[7],
                        HP_FEATURES[10],
                        'weather_error',
                    ]
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        print(
            'NaN features: '
            + ', '.join(
                k for k in MODEL_FEATURES
                if not np.isfinite(features[k])
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # RANDOM-FOREST MATRIX
    # ------------------------------------------------------------------
    # The CSV handed to the ML / Random Forest pipeline contains ONLY:
    #   - mp_id : identifier, kept only to join labels / trace predictions
    #   - the 30 quantitative MODEL_FEATURES
    #
    # Location and diagnostic metadata such as PLZ, Ort, Kanton,
    # weather_station and weather_error are deliberately excluded from the
    # ML matrix. This prevents the model from learning geography or string
    # metadata instead of the physical heat-pump signatures.
    full = pd.DataFrame(rows)

    model_columns = ['mp_id'] + MODEL_FEATURES
    model_table = full.reindex(columns=model_columns).copy()

    # MP ID is an identifier, not a numerical model feature.
    model_table['mp_id'] = model_table['mp_id'].astype(str)

    # Guarantee that every actual model feature is quantitative.
    for column in MODEL_FEATURES:
        model_table[column] = pd.to_numeric(model_table[column], errors='coerce')

    # Treat +/-inf as missing, same as NaN.
    model_table[MODEL_FEATURES] = model_table[MODEL_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # NaN REMOVAL — same convention as pv20/pv25/ev20
    # ------------------------------------------------------------------
    # Before: any column with a single NaN was dropped for ALL houses, which
    # on a big run wipes out most HP features. Now we fill per house like the
    # other scripts, and only drop a column if no house has a value at all.
    # --keep-nan writes the raw matrix untouched (for debugging).
    missing_counts = model_table[MODEL_FEATURES].isna().sum()

    if args.keep_nan:
        kept_features = list(MODEL_FEATURES)
        dropped_features = []
    else:
        filled = fill_missing_features(model_table.set_index('mp_id')[MODEL_FEATURES])
        kept_features = list(filled.columns)
        dropped_features = [c for c in MODEL_FEATURES if c not in kept_features]
        if not kept_features:
            raise RuntimeError('Every model feature is NaN for every house.')
        model_table = filled.reset_index()

        # Hard check: the RF must never see a NaN/inf.
        if not np.isfinite(model_table[kept_features].to_numpy(dtype=float)).all():
            raise RuntimeError('Model matrix still has non-finite values after filling.')

    model_table = model_table[['mp_id'] + kept_features]
    model_table.to_csv(args.output, index=False)

    filled_report = {c: int(missing_counts[c]) for c in kept_features if missing_counts[c] > 0}
    print(
        f'Model matrix: {len(kept_features)}/{len(MODEL_FEATURES)} features; '
        f'filled NaNs in {len(filled_report) if not args.keep_nan else 0} columns; '
        f'dropped {len(dropped_features)} all-NaN columns.',
        flush=True,
    )
    if dropped_features:
        print('Dropped (all-NaN) features: ' + ', '.join(dropped_features), flush=True)

    # Keep diagnostics in a separate file for debugging / audit only.
    # This file must NOT be passed to the Random Forest as X.
    diagnostic_columns = [
        column for column in full.columns
        if column not in MODEL_FEATURES and column != 'mp_id'
    ]
    if diagnostic_columns:
        diagnostic_output = args.output.with_name(
            args.output.stem + '_diagnostics.csv'
        )
        full[['mp_id'] + diagnostic_columns].to_csv(
            diagnostic_output,
            index=False,
        )
        print(f'Wrote diagnostics to {diagnostic_output}', flush=True)

    # Exact ordered list of the features ACTUALLY present in the no-NaN
    # Random-Forest matrix.
    args.output.with_suffix('.features.json').write_text(
        json.dumps(kept_features, indent=2)
    )

    # Audit file: how many houses had each feature missing before filling
    # (dropped = no house had it at all).
    n_houses = len(model_table)
    nan_report = {
        column: {
            'missing_count': int(missing_counts[column]),
            'missing_fraction': float(missing_counts[column] / n_houses) if n_houses else 0.0,
            'action': 'dropped' if column in dropped_features
                      else ('kept_nan' if args.keep_nan else 'filled'),
        }
        for column in MODEL_FEATURES if missing_counts[column] > 0
    }
    args.output.with_suffix('.nan_report.json').write_text(
        json.dumps(nan_report, indent=2)
    )

    print(
        f'Wrote {args.output}; {len(rows)} houses; '
        f'columns = mp_id + {len(kept_features)} features; '
        f'{"NaNs kept" if args.keep_nan else "0 NaN"}; source = {source_description}; '
        f'runtime {time.perf_counter()-start:.2f}s',
        flush=True,
    )

if __name__ == '__main__':
    main()