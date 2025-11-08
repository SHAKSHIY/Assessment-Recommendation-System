from selenium import webdriver 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from bs4 import BeautifulSoup
import time
import csv
import re

# Setup paths and URLs
chromedriver_path = r"C:\Users\shaks\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
base_url = "https://www.shl.com/solutions/products/product-catalog/?start={}&type=1"
detail_base = "https://www.shl.com"

# Headless Chrome options
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service, options=options)

# Helper function: Retry loading a URL
def safe_get(driver, url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            driver.get(url)
            return True
        except WebDriverException as e:
            print(f"🌐 Retry {attempt+1}/{retries} for {url} due to: {e}")
            time.sleep(delay)
    return False

data = []

# Loop through 32 pages (12 per page → 0 to 372 with step 12)
for start in range(0, 384, 12):
    url = base_url.format(start)
    print(f"📄 Scraping {url} ...")
    
    if not safe_get(driver, url):
        print(f"❌ Failed to load page: {url}. Skipping...")
        continue

    try:
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'tr[data-entity-id]')))
    except TimeoutException as e:
        print(f"⚠️ Timeout while waiting for page elements at {url}: {e}")
        continue

    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    rows = soup.select('tr[data-entity-id]')

    if not rows:
        print("No rows found, ending pagination loop.")
        break  # No more results

    for row in rows:
        try:
            # Assessment Name and Link
            title_tag = row.select_one('td.custom__table-heading__title a')
            name = title_tag.get_text(strip=True)
            link = detail_base + title_tag['href']

            # General TDs (Remote and Adaptive)
            general_tds = row.find_all("td", class_="custom__table-heading__general")
            
            remote = "No"
            adaptive = "No"
            
            if len(general_tds) > 0:
                remote_span = general_tds[0].find("span", class_=lambda x: x and "-yes" in x)
                if remote_span:
                    remote = "Yes"
            if len(general_tds) > 1:
                adaptive_span = general_tds[1].find("span", class_=lambda x: x and "-yes" in x)
                if adaptive_span:
                    adaptive = "Yes"

            # Tags (Test Type fallback)
            tag_elements = row.select('td.product-catalogue__keys span.product-catalogue__key')
            tags = [tag.get_text(strip=True) for tag in tag_elements]
            tags_str = ', '.join(tags) if tags else "N/A"

            # Duration and Test Type from detail page
            duration = "N/A"
            test_type = "N/A"

            if not safe_get(driver, link):
                print(f"❌ Failed to load detail page: {link}. Skipping assessment: {name}")
                continue

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".product-catalogue-training-calendar__row"))
                )
            except TimeoutException as e:
                print(f"⚠️ Timeout waiting for detail elements on {link}: {e}")
                continue

            time.sleep(1)
            detail_soup = BeautifulSoup(driver.page_source, 'html.parser')
            detail_block = detail_soup.select_one('.product-catalogue-training-calendar__row')

            # Extract duration
            duration_elem = detail_soup.find('p', string=re.compile("Approximate Completion Time in minutes"))
            if duration_elem:
                match = re.search(r"(\d+)", duration_elem.get_text())
                if match:
                    duration = f"{match.group(1)} minutes"

            # Extract test type
            if detail_block:
                test_type_container = detail_block.find('p', string=re.compile("Test Type:"))
                if test_type_container:
                    test_spans = test_type_container.find_all('span', class_='product-catalogue__key')
                    test_type = ', '.join(span.get_text(strip=True) for span in test_spans)

            if test_type == "N/A" or not test_type.strip():
                test_type = tags_str

            data.append([name, link, remote, adaptive, duration, test_type])
            print(f"✅ Fetched: {name}")

        except Exception as e:
            print(f"❌ Error fetching row: {e}")
            continue

driver.quit()

# Save to CSV
with open('shl_type1_detailed_catalog.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Assessment Name', 'URL', 'Remote Testing', 'Adaptive/IRT', 'Duration', 'Test Type'])
    writer.writerows(data)

print(f"\n✅ Done! Saved {len(data)} assessments to shl_type1_detailed_catalog.csv")
