import matplotlib
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict
from dateutil.parser import parse as parse_date
from geopy.geocoders import GoogleV3
from invisibleroads_macros.disk import make_enumerated_folder_for, make_folder
from invisibleroads_macros.log import format_summary
from matplotlib import pyplot as plt
from os.path import join
from pandas import DataFrame, Period, read_pickle
from pysal.cg.kdtree import Arc_KDTree
from pysal.cg.sphere import RADIUS_EARTH_KM
from sklearn.linear_model import LinearRegression


matplotlib.use('Agg')
EARTH_RADIUS_IN_METERS = RADIUS_EARTH_KM * 1000
GEOCODE = GoogleV3().geocode


def run(
        target_folder, target_date, from_date, to_date,
        search_address, search_radius_in_meters):
    t = read_pickle(join('datasets', 'nyc-traffic-injury-with-period.pkl'))
    t = _filter_by_dates(t, from_date, to_date)
    t = _filter_by_address(t, search_address, search_radius_in_meters)
    d = []
    if target_date:
        d.extend(_save_forecast_nyc_traffic_injury_table(
            target_folder, target_date, t))
    if 'Distance' in t.columns:
        d.extend(_save_nearby_nyc_traffic_injury_table(target_folder, t))
    d.extend(_save_worst_nyc_traffic_injury_tables(target_folder, t))
    d.extend(_plot_nyc_traffic_injury_summary_by_year(target_folder, t))
    d.extend(_plot_nyc_traffic_injury_summary_by_month(target_folder, t))
    return d


def _filter_by_dates(t, from_date, to_date):
    print('Filtering by dates...')
    if from_date:
        period = Period('%d-%02d' % (from_date.year, from_date.month))
        period_min = min(t.index)
        if period < period_min:
            period = period_min
        t = t[period:]
    if to_date:
        period = Period('%d-%02d' % (to_date.year, to_date.month))
        period_max = max(t.index)
        if period_max < period:
            period = period_max
        t = t[:period]
    return t


def _filter_by_address(t, search_address, search_radius_in_meters):
    print('Filtering by address...')
    if not search_address:
        return t
    search_location = GEOCODE(search_address)
    search_xy = search_location.longitude, search_location.latitude
    nyc_traffic_injury_tree = Arc_KDTree(
        t[['Longitude', 'Latitude']].values, radius=EARTH_RADIUS_IN_METERS)
    nyc_traffic_injury_count = nyc_traffic_injury_tree.n
    d = {}
    if search_radius_in_meters:
        d['distance_upper_bound'] = search_radius_in_meters
    distances, indices = nyc_traffic_injury_tree.query(
        search_xy, k=nyc_traffic_injury_tree.n, **d)
    t = t.reset_index()
    t = t.ix[indices[indices != nyc_traffic_injury_count]]
    t = t.set_index('Period')
    t['Distance'] = distances[distances != np.inf]
    return t


def _save_forecast_nyc_traffic_injury_table(target_folder, target_date, t):
    print('Forecasting...')
    target_year, target_month = target_date.year, target_date.month
    t = t.reset_index()
    bad_location_table = t.groupby([
        'Longitude', 'Latitude',
    ]).filter(lambda x: len(x) > 2)
    injury_types = ['Total', 'Pedestrian', 'Bike', 'Vehicle']
    rows = []
    for xy, local_injury_table in bad_location_table.groupby([
            'Longitude', 'Latitude']):
        local_injury_table = local_injury_table[['Period'] + injury_types]
        resampled_local_injury_table = local_injury_table.set_index(
            'Period').resample('M', np.sum).fillna(0).reset_index()
        xs = [x.to_timestamp().value for x in resampled_local_injury_table[
            'Period']]
        predicted_injury_counts = []
        for injury_type in injury_types:
            local_model = LinearRegression()
            local_model.fit(
                [[x] for x in xs],
                resampled_local_injury_table[injury_type].values)
            y = local_model.predict(Period(
                year=target_year, month=target_month, freq='M',
            ).to_timestamp().value)[0]
            predicted_injury_counts.append(y if y > 0 else 0)
        rows.append([
            xy[0], xy[1], target_year, target_month] + predicted_injury_counts)
    forecast_nyc_traffic_injury_table = DataFrame(rows, columns=[
        'Longitude', 'Latitude', 'Year', 'Month',
    ] + injury_types).sort(injury_types, ascending=False)
    forecast_nyc_traffic_injury_table['FillReds'] = forecast_nyc_traffic_injury_table['Total']  # noqa
    forecast_nyc_traffic_injury_table['RadiusInMetersRange10-50'] = forecast_nyc_traffic_injury_table['Total']  # noqa
    target_path = join(target_folder, 'forecast-nyc-traffic-injury.csv')
    forecast_nyc_traffic_injury_table.to_csv(target_path, index=False)
    return [('forecast_nyc_traffic_injury_geotable_path', target_path)]


def _save_nearby_nyc_traffic_injury_table(target_folder, t):
    print('Saving nearby...')
    t = t.sort([
        'Distance', 'Year', 'Month', 'Total',
    ], ascending=[True, False, False, False])
    t['FillBlues'] = t['Total']
    t['RadiusInMetersRange10-50'] = t['Total']
    target_path = join(target_folder, 'nearby-nyc-traffic-injury.csv')
    t.to_csv(target_path, index=False)
    return [('nearby_nyc_traffic_injury_geotable_path', target_path)]


def _save_worst_nyc_traffic_injury_tables(target_folder, t):
    print('Saving worst...')
    t = t.groupby([
        'Longitude', 'Latitude',
    ]).aggregate(OrderedDict([
        ('Total', np.sum),
        ('Pedestrian', np.sum),
        ('Bike', np.sum),
        ('Vehicle', np.sum),
    ])).reset_index()
    name_value_packs = []
    for injury_type in 'Pedestrian', 'Bike', 'Vehicle':
        x = t.sort(injury_type, ascending=False)
        x['FillBlues'] = x[injury_type]
        x['RadiusInMetersRange10-50'] = x[injury_type]
        injury_type_lower = injury_type.lower()
        target_path = join(
            target_folder,
            'worst-nyc-traffic-injury-%s.csv' % injury_type_lower)
        x.to_csv(target_path, index=False)
        name_value_packs.append((
            'worst_nyc_traffic_injury_%s_geotable_path' % injury_type_lower,
            target_path))
    return name_value_packs


def _plot_nyc_traffic_injury_summary_by_year(
        target_folder, nyc_traffic_injury_with_period_table):
    print('Plotting summary by year...')
    nyc_traffic_injury_by_year_table = nyc_traffic_injury_with_period_table[[
        'Total', 'Pedestrian', 'Bike', 'Vehicle']].resample('A', np.sum)
    fig = plt.figure()
    years = [x.year for x in nyc_traffic_injury_by_year_table.index]
    for index, (injury_type, color) in enumerate([
        ('Total', 'k'),
        ('Pedestrian', 'r'),
        ('Bike', 'g'),
        ('Vehicle', 'b'),
    ], 1):
        ax = plt.subplot(220 + index)
        ax.plot(years, nyc_traffic_injury_by_year_table[
            injury_type].values, linewidth=2, color=color)
        ax.set_title(injury_type)
        ax.set_xticklabels(years, rotation=45)
    fig.tight_layout()
    target_path = join(
        target_folder, 'selected_nyc_traffic_injury_summary_by_year.jpg')
    fig.savefig(target_path)
    return [('nyc_traffic_injury_summary_by_year_image_path', target_path)]


def _plot_nyc_traffic_injury_summary_by_month(target_folder, t):
    print('Plotting summary by month...')
    nyc_traffic_injury_by_month_table = t.groupby([
        'Year', 'Month',
    ]).aggregate(OrderedDict([
        ('Pedestrian', np.sum),
        ('Bike', np.sum),
        ('Vehicle', np.sum),
    ]))
    fig = plt.figure()
    for index, (injury_type, color) in enumerate([
        ('Pedestrian', 'r'),
        ('Bike', 'g'),
        ('Vehicle', 'b'),
    ], 1):
        ax = plt.subplot(130 + index)
        ax = nyc_traffic_injury_by_month_table[[injury_type]].plot(
            ax=ax,
            kind='barh',
            color=color,
            figsize=(7.5, 0.25 * len(nyc_traffic_injury_by_month_table)),
            width=1.01,
            edgecolor='none')
        ax.set_title(injury_type)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().get_label().set_visible(False)
        ax.set_yticklabels([
            '%d-%02d' % x for x in nyc_traffic_injury_by_month_table.index])
        ax.legend().remove()
    fig.tight_layout()
    target_path = join(
        target_folder,
        'selected_nyc_traffic_injury_summary_by_month.jpg')
    fig.savefig(target_path)
    return [('nyc_traffic_injury_summary_by_month_image_path', target_path)]


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        '--target_folder',
        metavar='FOLDER', type=make_folder)
    argument_parser.add_argument(
        '--target_date',
        metavar='DATE', type=parse_date)

    argument_parser.add_argument(
        '--from_date',
        metavar='DATE', type=parse_date)
    argument_parser.add_argument(
        '--to_date',
        metavar='DATE', type=parse_date)

    argument_parser.add_argument(
        '--search_address',
        metavar='ADDRESS')
    argument_parser.add_argument(
        '--search_radius_in_meters',
        metavar='RADIUS', type=float)

    args = argument_parser.parse_args()
    d = run(
        args.target_folder or make_enumerated_folder_for(__file__),
        args.target_date,
        args.from_date,
        args.to_date,
        args.search_address,
        args.search_radius_in_meters)
    print(format_summary(d))
