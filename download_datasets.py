from __future__ import print_function


import fiona
import requests
from argparse import ArgumentParser
from invisibleroads_macros.disk import make_folder
from invisibleroads_macros.log import format_summary
from os.path import abspath, dirname, getsize, join
from pandas import DataFrame, Period


def download_datasets(target_folder):
    print('Downloading...')
    source_folder = 'http://www.nyc.gov/html/dot/downloads/misc'
    nyc_traffic_injury_shapefile_path = download(
        join(target_folder, 'nyc-traffic-injury.shp.zip'),
        join(source_folder, 'injury_all_monthly_shapefile.zip'))

    print('Loading...')
    nyc_traffic_injury_table = prepare_nyc_traffic_injury_table(
        nyc_traffic_injury_shapefile_path)
    nyc_traffic_injury_table_path = \
        join(target_folder, 'nyc-traffic-injury.msg')
    nyc_traffic_injury_table.to_msgpack(
        nyc_traffic_injury_table_path, compress='blosc')

    print('Converting... (please be patient for more than 100 seconds)')
    nyc_traffic_injury_with_period_table = nyc_traffic_injury_table.apply(
        _add_period, axis=1)
    nyc_traffic_injury_with_period_table.set_index('Period', inplace=True)
    nyc_traffic_injury_with_period_table_path = \
        join(target_folder, 'nyc-traffic-injury-with-period.pkl')
    nyc_traffic_injury_with_period_table.to_pickle(
        nyc_traffic_injury_with_period_table_path)
    return [(
        'nyc_traffic_injury_table_path',
        nyc_traffic_injury_table_path), (
        'nyc_traffic_injury_with_period_table_path',
        nyc_traffic_injury_with_period_table_path)]


def download(target_path, source_url):
    print(source_url, end=' ')
    response = requests.get(source_url)
    open(target_path, 'w').write(response.content)
    print('{:,}'.format(getsize(target_path)))
    return target_path


def prepare_nyc_traffic_injury_table(shapefile_path):
    collection = fiona.open('/', vfs='zip://' + shapefile_path)
    rows, indices = [], []
    for d in collection:
        indices.append(int(d['id']))
        longitude, latitude = map(float, d['geometry']['coordinates'])
        properties = d['properties']
        year, month = int(properties['YR']), int(properties['MN'])
        total_count = int(properties['Injuries'])
        pedestrian_count = int(properties['PedInjurie'])
        bike_count = int(properties['BikeInjuri'])
        vehicle_count = int(properties['MVOInjurie'])
        rows.append([
            longitude, latitude, year, month,
            total_count, pedestrian_count, bike_count, vehicle_count,
        ])
    return DataFrame(rows, index=indices, columns=[
        'Longitude', 'Latitude', 'Year', 'Month',
        'Total', 'Pedestrian', 'Bike', 'Vehicle',
    ])


def _add_period(row):
    row['Period'] = Period(year=row['Year'], month=row['Month'], freq='M')
    return row


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        '--target_folder',
        metavar='FOLDER', type=make_folder,
        default=join(dirname(abspath(__file__)), 'datasets'))
    args = argument_parser.parse_args()
    d = download_datasets(args.target_folder)
    print(format_summary(d))