from urllib.parse import parse_qs, urlparse
import googlesearch
from googlesearch import url_parameters, url_home, url_next_page, url_next_page_num, url_search, url_search_num, get_page
import pandas as pd
import datetime
from tqdm import tqdm
import os
try:
    from bs4 import BeautifulSoup
    is_bs4 = True
except ImportError:
    from BeautifulSoup import BeautifulSoup
    is_bs4 = False
import time

import sys
if sys.version_info[0] > 2:
    from urllib.parse import quote_plus
else:
    from urllib import quote_plus
import json
import argparse
import re


url_to_html_cache = {}
if os.path.exists("cache/links_to_html_cache.jsonl"):
    with open("cache/links_to_html_cache.jsonl") as f:
        for line in f:
            line = json.loads(line)
            url = line["url"]
            html = line["html"]
            url_to_html_cache[url] = html
ORIGIN_DATE = datetime.datetime.today()


def get_google_url(query, tld='com', lang='en', tbs='0', safe='off', num=10, start=0,
           stop=None, pause=2.0, country='', extra_params=None,
           user_agent=None, verify_ssl=True):
    """
    Search the given query string using Google.

    return: url (str) for the search on Google
    """
    # Set of hashes for the results found.
    # This is used to avoid repeated results.
    hashes = set()

    # Count the number of links yielded.
    count = 0

    # Prepare the search string.
    query = quote_plus(query)

    # If no extra_params is given, create an empty dictionary.
    # We should avoid using an empty dictionary as a default value
    # in a function parameter in Python.
    if not extra_params:
        extra_params = {}

    # Check extra_params for overlapping.
    for builtin_param in url_parameters:
        if builtin_param in extra_params.keys():
            raise ValueError(
                'GET parameter "%s" is overlapping with \
                the built-in GET parameter',
                builtin_param
            )

    # Prepare the URL of the first request.
    if start:
        if num == 10:
            url = url_next_page % vars()
        else:
            url = url_next_page_num % vars()
    else:
        if num == 10:
            url = url_search % vars()
        else:
            url = url_search_num % vars()

    # Append extra GET parameters to the URL.
    # This is done on every iteration because we're
    # rebuilding the entire URL at the end of this loop.
    for k, v in extra_params.items():
        k = quote_plus(k)
        v = quote_plus(v)
        url = url + ('&%s=%s' % (k, v))
    return url


# Filter links found in the Google result pages HTML code.
# Returns None if the link doesn't yield a valid result.
def filter_result(link):
    try:

        # Decode hidden URLs.
        if link.startswith('/url?'):
            o = urlparse(link, 'http')
            link = parse_qs(o.query)['q'][0]

        # Valid results are absolute URLs not pointing to a Google domain,
        # like images.google.com or googleusercontent.com for example.
        # TODO this could be improved!
        o = urlparse(link, 'http')
        if o.netloc and 'google' not in o.netloc:
            return link

    # On error, return None.
    except Exception:
        pass


# Returns a generator that yields URLs.
def search(query, tld='com', lang='en', tbs='0', safe='off', num=10, start=0,
           stop=None, pause=2.0, country='', extra_params=None,
           user_agent=None, verify_ssl=True):
    # Set of hashes for the results found.
    # This is used to avoid repeated results.
    hashes = set()

    # Count the number of links yielded.
    count = 0

    # Prepare the search string.
    query = quote_plus(query)

    # If no extra_params is given, create an empty dictionary.
    # We should avoid using an empty dictionary as a default value
    # in a function parameter in Python.
    if not extra_params:
        extra_params = {}

    # Check extra_params for overlapping.
    for builtin_param in url_parameters:
        if builtin_param in extra_params.keys():
            raise ValueError(
                'GET parameter "%s" is overlapping with \
                the built-in GET parameter',
                builtin_param
            )

    # Grab the cookie from the home page.
    get_page(url_home % vars(), user_agent, verify_ssl)

    # Prepare the URL of the first request.
    if start:
        if num == 10:
            url = url_next_page % vars()
        else:
            url = url_next_page_num % vars()
    else:
        if num == 10:
            url = url_search % vars()
        else:
            url = url_search_num % vars()

    # Loop until we reach the maximum result, if any (otherwise, loop forever).
    while not stop or count < stop:

        # Remeber last count to detect the end of results.
        last_count = count

        # Append extra GET parameters to the URL.
        # This is done on every iteration because we're
        # rebuilding the entire URL at the end of this loop.
        for k, v in extra_params.items():
            k = quote_plus(k)
            v = quote_plus(v)
            url = url + ('&%s=%s' % (k, v))


        if url in url_to_html_cache:
            html = url_to_html_cache[url]
        else:
            # Sleep between requests.
            # Keeps Google from banning you for making too many requests.
            time.sleep(pause)
            # Request the Google Search results page.
            html = get_page(url, user_agent, verify_ssl)
            url_to_html_cache[url] = html
            with open("cache/links_to_html_cache.jsonl", "a") as f:
                f.write(json.dumps({"url": url, "html": html.decode('utf-8')}) + "\n")

        # Parse the response and get every anchored URL.
        if is_bs4:
            soup = BeautifulSoup(html, 'html.parser')
        else:
            soup = BeautifulSoup(html)
        try:
            anchors = soup.find(id='search').findAll('a')
            # Sometimes (depending on the User-agent) there is
            # no id "search" in html response...
        except AttributeError:
            # Remove links of the top bar.
            gbar = soup.find(id='gbar')
            if gbar:
                gbar.clear()
            anchors = soup.findAll('a')
            # anchors = soup.find(role='heading')

        # Process every anchored URL.
        for a in anchors:
            # Get the URL from the anchor tag.
            try:
                link = a['href']
            except KeyError:
                continue

            # Filter invalid links and links pointing to Google itself.
            link = filter_result(link)
            if not link:
                continue

            # Discard repeated results.
            h = hash(link)
            if h in hashes:
                continue
            hashes.add(h)

            # for child in a.children:
            #     # get 
            # a
            #   div
            #       div [img]
            #       div [text]
            #           div [source]
            #           <div> title </div>
            #           div [description]
            #           span
            #           div [date]
            #               <span> date </span>
            # title_div = a.find_all('div')[1]
            # title = title_div.find('div').text
            # date_div = a.find_all('div')[2]
            # date = date_div.find('span').text

            title_div = a.find('div')
            # Extract the title
            title = title_div.find('span').text
            date_div = a.find_all('div')[2]
            # Extract the date
            date = date_div.find_all('span')[-1].text

            # Yield the result.
            yield link, title, date, url

            # Increase the results counter.
            # If we reached the limit, stop.
            count += 1
            if stop and count >= stop:
                return

        # End if there are no more results.
        # XXX TODO review this logic, not sure if this is still true!
        if last_count == count:
            break

        # Prepare the URL for the next request.
        start += num
        if num == 10:
            url = url_next_page % vars()
        else:
            url = url_next_page_num % vars()


def extract_queries(query, tbs):
    results = []
    # google_results = googlesearch.search(
    google_results = search(
        query, num=10, stop=10,
        tbs=tbs,
        extra_params={"tbm": "nws"},
    )
    search_url = None
    for link, title, date, search_url in google_results:
        # date is of form "MM months ago"
        if "month" in date:
            # xx month(s) ago
            assert re.match(r"\d+ month(s)? ago", date)
            num_months = int(date.split()[0])
        else:
            try:
                assert re.match(r"\d+ week(s)? ago", date) or re.match(r"\d+ day(s)? ago", date) or re.match(r"\d+ hour(s)? ago", date) or re.match(r"\d+ minute(s)? ago", date) or re.match(r"\d+ second(s)? ago", date)
            except:
                print(date)
            num_months = 0
        # compute year and month
        year = ORIGIN_DATE.year - (num_months // 12)
        remaining_months = num_months % 12
        month = ORIGIN_DATE.month - remaining_months
        if month <= 0:
            month += 12
            year -= 1
        date = datetime.datetime(year, month, 1).strftime("%m/%Y")
        results.append({"link": link, "title": title, "date": date})
    return results, search_url


def main(source_csv, target_csv):
    links_dict = {
        "subjectLabel": [],
        "propertyLabel": [],
        "objectLabel": [],
        "startDate": [],
        "endDate": [],
        "searchURL": [],
        "endDateSus": [],
    }
    unfiltered_target_csv = target_csv.replace(".csv", "_unfiltered.csv")
    if os.path.exists(unfiltered_target_csv):
        existing_links = pd.read_csv(unfiltered_target_csv)
    else:
        existing_links = pd.DataFrame.from_dict(links_dict)
    
    for i in range(10):
        links_dict["links_" + str(i)] = []
    with open(source_csv) as f:
        results = pd.read_csv(f)
        new_results = pd.DataFrame(columns=results.columns)
        # find adjacent entries with same (subject, property, object)
        for i, result in results.iterrows():
            if i < 1:
                continue
            if result["subjectLabel"] == results.iloc[i - 1]["subjectLabel"] and result["propertyLabel"] == results.iloc[i - 1]["propertyLabel"] and result["objectLabel"] == results.iloc[i - 1]["objectLabel"]:
                new_results["endDate"][new_results.shape[0]-1] = result["endDate"]
            else:
                new_results.loc[new_results.shape[0]] = result
        results = new_results

        num_results = 0
        for i, result in tqdm(results.iterrows(), total=results.shape[0]):
            links_dict["subjectLabel"].append(result["subjectLabel"])
            links_dict["propertyLabel"].append(result["propertyLabel"])
            links_dict["objectLabel"].append(result["objectLabel"])
            links_dict["startDate"].append(result["startDate"])
            links_dict["endDate"].append(result["endDate"])
            if result["objectLabel"] == result["objectLabel"]:
                query = result["subjectLabel"] + " " + result["propertyLabel"] + " " + result["objectLabel"]
            else:
                query = result["subjectLabel"] + " " + result["propertyLabel"]

            if result["startDate"] == result["startDate"] and not result["startDate"].startswith("http://www.wikidata.org/.well-known/genid/"):
                start_date = datetime.datetime.strptime(result["startDate"], "%Y-%m-%dT%H:%M:%SZ")
            else:
                start_date = datetime.datetime.strptime("1970-01-01", "%Y-%m-%d")
            if result["endDate"] == result["endDate"] and not result["endDate"].startswith("http://www.wikidata.org/.well-known/genid/"):
                end_date = datetime.datetime.strptime(result["endDate"], "%Y-%m-%dT%H:%M:%SZ")
            else:
                end_date = datetime.datetime.now()
            if end_date == start_date:
                end_date = start_date + datetime.timedelta(days=365)

            other_subj_attr_rows = results[(
                results["subjectLabel"] == result["subjectLabel"]
            ) & (
                results["propertyLabel"] == result["propertyLabel"]
            ) & (
                results["objectLabel"] != result["objectLabel"]
            ) & (
                results["startDate"] > result["startDate"]
            )]
            if other_subj_attr_rows.shape[0] > 0:
                # get row in other_subj_attr_rows with min start date
                next_subj_attr_row = other_subj_attr_rows[other_subj_attr_rows["startDate"] == other_subj_attr_rows["startDate"].min()]
                links_dict["endDateSus"].append(((result["endDate"] != result["endDate"]) | (result["endDate"] > next_subj_attr_row["startDate"])).values[0])
            else:
                links_dict["endDateSus"].append(False)

            valid_rows = existing_links[(
                existing_links["subjectLabel"] == result["subjectLabel"]
            ) & (
                existing_links["propertyLabel"] == result["propertyLabel"]
            ) & (
                existing_links["objectLabel"] == result["objectLabel"]
            ) & (
                existing_links["startDate"] == result["startDate"]
            )]

            tbs = googlesearch.get_tbs(from_date=start_date, to_date=min(end_date, start_date + datetime.timedelta(days=90)))
            if len(valid_rows) > 0:
                # already have result
                for q in range(10):
                    links_dict[f"links_{q}"].append(valid_rows.iloc[0][f"links_{q}"])
                links_dict["searchURL"].append(valid_rows.iloc[0]["searchURL"])
            elif result["objectLabel"] != result["objectLabel"]:
                for q in range(10):
                    links_dict[f"links_{q}"].append(None)
                links_dict["searchURL"].append(get_google_url(
                    query, num=10, stop=10,
                    tbs=tbs,
                    extra_params={"tbm": "nws"},
                ))
            else:
                num_results += 1
                # get result
                query_results, search_url = extract_queries(query, tbs)
                if len(query_results) == 0:
                    # expand to original end date
                    tbs = googlesearch.get_tbs(from_date=start_date, to_date=end_date)
                    query_results, search_url = extract_queries(query, tbs)
                    for q in range(10):
                        links_dict[f"links_{q}"].append(query_results[q] if q < len(query_results) else None)
                    if search_url is None:
                        search_url = get_google_url(
                            query, num=10, stop=10,
                            tbs=tbs,
                            extra_params={"tbm": "nws"},
                        )
                    links_dict["searchURL"].append(search_url)
                else:
                    for q in range(10):
                        links_dict[f"links_{q}"].append(query_results[q] if q < len(query_results) else None)
                    links_dict["searchURL"].append(search_url)
            links_df = pd.DataFrame.from_dict(links_dict)
            with open(unfiltered_target_csv, "w") as f:
                links_df.to_csv(f, index=False)
        print(num_results)

    filtered_links_df = pd.DataFrame.from_dict(links_dict)
    for i, result in links_df.iterrows():
        if result["endDateSus"]:
            filtered_links_df = filtered_links_df.drop(i)
        elif result["links_0"] != result["links_0"]:
            filtered_links_df = filtered_links_df.drop(i)
    for i, result in filtered_links_df.iterrows():
        other_subj_attr_rows = filtered_links_df[(
            filtered_links_df["subjectLabel"] == result["subjectLabel"]
        ) & (
            filtered_links_df["propertyLabel"] == result["propertyLabel"]
        ) & (
            filtered_links_df["objectLabel"] != result["objectLabel"]
        )]
        if other_subj_attr_rows.shape[0] == 0:
            filtered_links_df = filtered_links_df.drop(i)
    with open(target_csv, "w") as f:
        filtered_links_df.to_csv(f, index=False)


if __name__ == "__main__":
    source_csv = "data/property_to_results.csv"
    target_csv = "data/property_to_results_links.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", type=str, default=source_csv)
    parser.add_argument("--target_csv", type=str, default=target_csv)
    main(**vars(parser.parse_args()))
