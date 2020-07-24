import requests
import datetime
from datetime import timedelta, date
from pathlib import Path
import time


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    # Overrides from the library to keep headers when redirected to or from the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if (original_parsed.hostname != redirect_parsed.hostname) and \
               redirect_parsed.hostname != self.AUTH_HOST and \
               original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return


username = 'jxiong21029'
password = 'WsthV68%22C*YZ'
session = SessionWithHeaderRedirection(username, password)

root = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.06'


start_time = time.time()
start_date = date(2005, 1, 1)
end_date = date(2020, 7, 1)
for d in date_range(start_date, end_date):
    url = f'{root}/{d.year}/{d.month:02}/3B-DAY-L.MS.MRG.3IMERG.{d.year}{d.month:02}{d.day:02}-S000000-E235959.V06.nc4'
    response = session.get(url, stream=True)
    if response:
        Path(f'D:/IMERG_DATA/{d.year}/{d.month:02}').mkdir(parents=True, exist_ok=True)
        filename = f'D:/IMERG_DATA/{d.year}/{d.month:02}/{d.day:02}.nc4'
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=None):
                f.write(chunk)
        elapsed = time.time() - start_time
        num_completed = (d - start_date).days + 1
        num_remaining = (end_date - d).days
        eta = datetime.datetime.now() + timedelta(seconds=num_remaining * elapsed / num_completed)
        print(f'Successfully downloaded data for {d}, ETA={eta}')
    else:
        print(f'Issues with downloading data for {d}')
