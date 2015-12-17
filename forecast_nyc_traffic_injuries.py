import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict
from dateutil.parser import parse as parse_date
from geopy.geocoders import GoogleV3
from invisibleroads_macros.disk import make_enumerated_folder_for, make_folder
from invisibleroads_macros.log import format_summary
from matplotlib import pyplot as plt
from os.path import join
from pandas import Period, read_pickle
from pysal.cg.kdtree import Arc_KDTree
from pysal.cg.sphere import RADIUS_EARTH_KM


EARTH_RADIUS_IN_METERS = RADIUS_EARTH_KM * 1000
GEOCODE = GoogleV3().geocode


def run(
        target_folder, target_date, from_date, to_date,
        search_address, search_radius_in_meters):
    t = read_pickle(join('datasets', 'nyc-traffic-injury-with-period.pkl'))
    t = _filter_by_dates(t, from_date, to_date)
    t = _filter_by_address(t, search_address, search_radius_in_meters)
    d = [(
        'recent_nyc_traffic_injury_table_path',
        _save_recent_nyc_traffic_injury_table(target_folder, t)), (
        'worst_nyc_traffic_injury_table_path',
        _save_worst_nyc_traffic_injury_table(target_folder, t)), (
        'nyc_traffic_injury_summary_by_year_image_path',
        _plot_nyc_traffic_injury_summary_by_year(target_folder, t)), (
        'nyc_traffic_injury_summary_by_month_image_path',
        _plot_nyc_traffic_injury_summary_by_month(target_folder, t))]
    if 'Distance' not in t.columns:
        return d
    d.insert(0, (
        'nearby_nyc_traffic_injury_table_path',
        _save_nearby_nyc_traffic_injury_table(target_folder, t)))
    return d


def _filter_by_dates(t, from_date, to_date):
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


def _save_nearby_nyc_traffic_injury_table(target_folder, t):
    target_path = join(target_folder, 'nearby-nyc-traffic-injury.csv')
    t.sort([
        'Distance', 'Year', 'Month', 'Total',
    ], ascending=[True, False, False, False]).to_csv(target_path, index=False)
    return target_path


def _save_recent_nyc_traffic_injury_table(target_folder, t):
    target_path = join(target_folder, 'recent-nyc-traffic-injury.csv')
    t.sort([
        'Year', 'Month', 'Total',
    ], ascending=False).to_csv(target_path, index=False)
    return target_path


def _save_worst_nyc_traffic_injury_table(target_folder, t):
    target_path = join(target_folder, 'worst-nyc-traffic-injury.csv')
    t.groupby([
        'Longitude', 'Latitude',
    ]).aggregate(OrderedDict([
        ('Total', np.sum),
        ('Pedestrian', np.sum),
        ('Bike', np.sum),
        ('Vehicle', np.sum),
    ])).sort([
        'Total', 'Pedestrian', 'Bike', 'Vehicle',
    ], ascending=False).reset_index().to_csv(target_path, index=False)
    return target_path


def _plot_nyc_traffic_injury_summary_by_year(
        target_folder, nyc_traffic_injury_with_period_table):
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
    return target_path


def _plot_nyc_traffic_injury_summary_by_month(target_folder, t):
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
    return target_path


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
