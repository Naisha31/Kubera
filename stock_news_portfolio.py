!pip install sentence-transformers
!apt-get install -y tesseract-ocr
!pip install pytesseract
!pip install wandb


import numpy as np
from sentence_transformers import SentenceTransformer, util

from PIL import Image
import pytesseract
import requests
import re
from datetime import datetime, timedelta
import pandas as pd

# Step 1: Extract text from the image
image_path = '/content/image2.jpg'
image = Image.open(image_path)
text = pytesseract.image_to_string(image)

# Step 2: Clean and parse the extracted text to identify the company names
# Print the extracted text for debugging purposes
#print("Extracted Text:\n", text)

company_names = [
    "Canara Bank", "State Bank of India", "ICICI Bank", "HDFC Bank", "Axis Bank",
    "Reliance Industries", "Tata Consultancy Services", "Infosys", "Hindustan Unilever",
    "ITC", "Kotak Mahindra Bank", "Bharti Airtel", "HCL Technologies", "Asian Paints",
    "Maruti Suzuki", "Larsen & Toubro", "Bajaj Finance", "Wipro", "Mahindra & Mahindra",
    "Nestle India", "UltraTech Cement", "SBI Life Insurance", "Power Grid Corporation",
    "Sun Pharmaceutical", "Titan Company", "HDFC Life Insurance", "Tech Mahindra",
    "IndusInd Bank", "Dr. Reddy's Laboratories", "Britannia Industries", "Divi's Laboratories",
    "NTPC", "JSW Steel", "Tata Steel", "Tata Motors", "Grasim Industries", "Bajaj Finserv",
    "Adani Green Energy", "Adani Ports", "Adani Enterprises", "Adani Transmission",
    "Tata Power", "Hindalco Industries", "Vedanta", "Eicher Motors", "Bharat Petroleum",
    "Indian Oil Corporation", "Oil and Natural Gas Corporation", "Coal India",
    "Hindustan Petroleum", "GAIL", "Zee Entertainment", "Pidilite Industries",
    "Shree Cement", "Godrej Consumer Products", "Hero MotoCorp", "Havells India",
    "Cipla", "Aurobindo Pharma", "Muthoot Finance", "Cholamandalam Investment",
    "SBI Cards and Payment Services", "Tata Consumer Products", "Dabur India",
    "ICICI Prudential Life Insurance", "ICICI Lombard General Insurance", "Siemens",
    "Bosch", "Page Industries", "Mahanagar Gas", "United Breweries", "ABB India",
    "Biocon", "Bajaj Auto", "Ambuja Cements", "ACC", "Colgate-Palmolive (India)",
    "Procter & Gamble Hygiene", "3M India", "Castrol India", "Torrent Pharmaceuticals",
    "SRF", "Lupin", "Cadila Healthcare", "Glenmark Pharmaceuticals", "Jubilant FoodWorks",
    "Motherson Sumi Systems", "Apollo Hospitals", "ICICI Securities", "Indraprastha Gas",
    "AU Small Finance Bank", "Bandhan Bank", "IDFC First Bank", "Yes Bank", "Federal Bank",
    "RBL Bank", "City Union Bank", "Bank of Baroda", "Punjab National Bank", "Suzlon Energy",
    "Ircon International", "SAIL", "BHEL", "IRFC", "Bharat Electronics",
    "Bharti Infratel", "Piramal Enterprises", "Godrej Properties", "Adani Total Gas",
    "MindTree", "Tata Chemicals", "DLF", "L&T Technology Services", "GMR Infrastructure",
    "JSW Energy", "Adani Wilmar", "Hindustan Zinc", "Mphasis", "Berger Paints",
    "Jindal Steel & Power", "Aditya Birla Capital", "Ruchi Soya Industries", "Tata Elxsi",
    "Astral Poly Technik", "Balkrishna Industries", "Max Financial Services", "Emami",
    "Blue Star", "Crompton Greaves Consumer Electricals", "Dixon Technologies",
    "Sundaram Finance", "Torrent Power", "Marico", "Havells India", "TVS Motor Company",
    "Amara Raja Batteries", "Escorts", "PVR", "ICICI Prudential", "Bata India",
    "Jindal Stainless", "Fortis Healthcare", "Indiabulls Housing Finance", "IRCTC",
    "Nippon Life India Asset Management", "Metropolis Healthcare", "SpiceJet",
    "Ujjivan Small Finance Bank", "Varun Beverages", "Route Mobile", "Trident", "LT Foods",
    "Welspun India", "V-Guard Industries", "Ruchi Soya", "Indus Towers", "Lupin Limited",
    "Avenue Supermarts", "Hindustan Copper", "Balaji Amines", "Jubilant Ingrevia",
    "Aarti Industries", "Crompton Greaves", "Narayana Hrudayalaya", "Manappuram Finance",
    "Granules India", "DCB Bank", "Ashok Leyland", "National Aluminium Company",
    "Hexaware Technologies", "Sundram Fasteners", "Alembic Pharmaceuticals",
    "Can Fin Homes", "Deepak Nitrite", "Alembic Ltd.", "Indoco Remedies", "KEI Industries",
    "KPIT Technologies", "Kalyan Jewellers", "Kirloskar Oil Engines", "KRBL",
    "L&T Finance Holdings", "L&T Infotech", "NCC Ltd.", "Navin Fluorine International",
    "Polycab India", "PSP Projects", "Rallis India", "Schaeffler India", "Sharda Cropchem",
    "Shilpa Medicare", "Sterlite Technologies", "Supreme Industries", "Thermax",
    "Thyrocare Technologies", "Timken India", "V-Mart Retail", "Vijaya Diagnostic Centre",
    "VIP Industries", "Voltas", "Wonderla Holidays", "Zydus Wellness", "IDBI Bank",
    "NLC India", "Prestige Estates", "Raymond", "Redington India", "SBI Mutual Fund",
    "Steel Authority of India", "Sun TV Network", "Tata Coffee", "Tata Communications",
    "Tata Investment Corporation", "Texmaco Rail & Engineering", "Thangamayil Jewellery",
    "Triveni Turbine", "UCO Bank", "Union Bank of India", "Usha Martin", "UTI Asset Management",
    "Vadilal Industries", "Vaibhav Global", "Vakrangee", "Venky's India", "Wockhardt",
    "Zensar Technologies", "Zodiac Clothing", "Zydus Cadila", "Shyam Metalics",
    "Borosil Renewables", "Polaris Consulting", "Sasken Technologies", "Capri Global Capital",
    "DCM Shriram", "TTK Prestige", "Gulf Oil Lubricants", "Welspun Corp", "Orient Cement",
    "Finolex Cables", "IIFL Wealth Management", "Mahindra Holidays", "GIC Housing Finance",
    "Rane Holdings", "IRB Infrastructure", "ITD Cementation", "Dalmia Bharat", "Alok Industries",
    "Ajanta Pharma", "Allcargo Logistics", "Alkem Laboratories", "Amara Raja Batteries",
    "Apar Industries", "Ashoka Buildcon", "Atul Ltd.", "Balkrishna Industries",
    "Bank of Maharashtra", "BASF India", "BEML", "Best Agrolife", "BGR Energy Systems",
    "Birlasoft", "Bliss GVS Pharma", "Bombay Burmah Trading", "Cams Services",
    "Caplin Point Laboratories", "Century Plyboards", "Chemplast Sanmar",
    "Cochin Shipyard", "CreditAccess Grameen", "CSB Bank", "DCM Shriram",
    "Delta Corp", "Dish TV India", "Dilip Buildcon", "Dhanuka Agritech",
    "Edelweiss Financial Services", "EID Parry", "Engineers India", "Everest Industries",
    "FDC Ltd.", "Future Consumer", "Glenmark Life Sciences", "Godrej Agrovet",
    "Goodricke Group", "GP Petroleums", "Gufic Biosciences", "Happiest Minds",
    "Hikal", "Hindustan Media Ventures", "HMT Ltd.", "HSIL", "ICRA",
    "Indiabulls Real Estate", "Indoco Remedies", "Intellect Design Arena",
    "Ion Exchange", "Jagran Prakashan", "Jain Irrigation Systems", "Jamna Auto",
    "Jayshree Tea", "Jindal Poly Films", "JK Lakshmi Cement", "JK Paper",
    "JK Tyre & Industries", "JMC Projects", "JM Financial", "Kabra Extrusiontechnik",
    "Kalpataru Power Transmission", "Kansai Nerolac Paints", "Kaveri Seed Company",
    "KDDL Ltd.", "Kirloskar Brothers", "Kolte Patil Developers", "KPR Mill",
    "KSB Pumps", "La Opala RG", "Lakshmi Machine Works", "Lasa Supergenerics",
    "Laxmi Organic Industries", "Lemon Tree Hotels", "Linde India", "Lux Industries",
    "Mangalore Refinery & Petrochemicals", "Marksans Pharma", "MAS Financial Services",
    "Mazagon Dock Shipbuilders", "Meghmani Organics", "Menon Bearings",
    "Mishra Dhatu Nigam", "MMTC Ltd.", "Monnet Ispat & Energy", "MOIL",
    "MTNL", "Muthoot Capital Services", "Nandan Denim", "Natco Pharma",
    "National Fertilizers", "NESCO Ltd.", "Nilkamal", "Nitin Spinners",
    "NLC India", "Nucleus Software Exports", "Olectra Greentech", "Oil India",
    "Orient Electric", "Orient Refractories", "PNB Housing Finance", "Ponni Sugars",
    "Praj Industries", "PTC India", "Quess Corp", "Quick Heal Technologies",
    "Radico Khaitan", "Rajesh Exports", "Ramco Cements", "Ramkrishna Forgings",
    "Rashtriya Chemicals & Fertilizers", "Redington India", "Reliance Capital",
    "RITES Ltd.", "RPG Life Sciences", "Sanghvi Movers", "Sanofi India",
    "Sarda Energy & Minerals", "Satin Creditcare Network", "Sequent Scientific",
    "Shree Renuka Sugars", "Siti Networks", "Snowman Logistics", "Sobha",
    "Solara Active Pharma", "Sonata Software", "South Indian Bank", "Spencer's Retail",
    "Sterling & Wilson Solar", "Subros", "Sudarshan Chemical", "Sunteck Realty",
    "Surya Roshni", "Swan Energy", "Syngene International", "Tamil Nadu Newsprint",
    "Tata Metaliks", "TCI Express", "TCS", "The India Cements", "Thyrocare",
    "Titan Biotech", "Torrent Gas", "Tribhovandas Bhimji Zaveri", "Triveni Engineering",
    "TTK Healthcare", "TV Today Network", "Ujjivan Financial Services", "Usha International",
    "UTL Industries", "Vascon Engineers", "Venus Remedies", "Vesuvius India",
    "Vimta Labs", "Vishal Fabrics", "Vivimed Labs", "VRL Logistics", "WABCO India",
    "Welspun Enterprises", "Wonderla Holidays", "Xelpmoc Design and Tech", "Zee Learn",
    "Zee Media Corporation", "Zydus Cadila", "Zydus Wellness", "Zydus Lifesciences",
    "Aegis Logistics", "Ahmedabad Steelcraft", "Air India", "Airports Authority of India",
    "Ajmera Realty", "Albert David", "Alfred Herbert", "Alstom India", "Aluminium Industries",
    "Amines & Plasticizers", "Andhra Bank", "Andhra Cements", "Anup Engineering", "Apar Industries",
    "Apollo Tyres", "Arvind Ltd.", "Ashiana Housing", "Asian Granito", "Associated Alcohols & Breweries",
    "AstraZeneca Pharma India", "Autoline Industries", "Avanti Feeds", "Aviation Industry Corp",
    "Baba Arts", "Balasore Alloys", "Ballarpur Industries", "Bangalore Fort Farms", "Bang Overseas",
    "Bank of Rajasthan", "Bartronics India", "Bata India", "Bay Commercial Bank", "BDH Industries",
    "Bee Electronics Machines", "Benares Hotels", "Bengal & Assam Company", "Bengal Tea & Fabrics",
    "Berger Paints India", "Betex India", "BF Investment", "BGR Energy Systems", "Bhagwati Autocast",
    "Bhageria Industries", "Bhandari Hosiery Exports", "Bhansali Engineering Polymers", "Bharat Forge",
    "Bharat Gears", "Bharat Hotels", "Bharat Rasayan", "Bharat Road Network", "Bharat Seats",
    "Bharat Wire Ropes", "Bhartiya International", "Bhilwara Spinners", "Bhima Jewellers",
    "Bhoomi Infrastructures", "Bhoomika Industries", "Binani Industries", "Birla Cable",
    "Birla Corporation", "Birla Precision Technologies", "Borosil Glass Works", "Boston International",
    "BPL Limited", "BPL Telecom", "Brady & Morris", "Bright Brothers", "Britannia Industries",
    "Brooks Laboratories", "BS Limited", "BSEL Infrastructure", "Budget Hotels", "Bulls Eye Ventures",
    "Butterfly Gandhimathi Appliances", "C Mahendra Exports", "Cairn India", "Cals Refineries",
    "Cambridge Technology", "Camlin Fine Sciences", "Canara HSBL", "Capri Global", "Career Point",
    "Cargill India", "Carlyle India Advisors", "Castrol India", "Central Bank of India", "Central Depository Services",
    "Century Enka", "Century Textiles", "Ceat Limited", "Cera Sanitaryware", "CESC Ventures",
    "Chambal Fertilizers", "Chembond Chemicals", "Chemplast Sanmar", "Chennai Meenakshi Multispeciality Hospital",
    "Chennai Petroleum", "Choksi Laboratories", "Cholamandalam Financial Holdings", "Cimmco", "Cipla Limited",
    "Citi Union Finance", "City Online Services", "Clariant Chemicals", "Classic Diamonds", "CL Educate",
    "Clean Science and Technology", "Coastal Corporation", "Cochin Malabar Estates", "Coffee Day Enterprises",
    "Colgate Palmolive", "Colortek India", "Comed Chemicals", "Commercial Engineers", "Commercial Syn Bags",
    "Computer Age Management", "Concept Pharmaceuticals", "Confidence Petroleum", "Container Corporation",
    "Continental Engines", "Control Print", "Coral Laboratories", "Coromandel International", "Cosmo Ferrites",
    "Cranes Software", "Credent Global Finance", "Creative Castings", "Crisil Limited", "CRISIL SME Track",
    "Crompton Greaves Consumer", "CSL Finance", "Cupid Limited", "Cushman & Wakefield", "Cyient Limited",
    "Dabur India", "Dai Ichi Karkaria", "Dalal Street Investments", "Dalmia Bharat Sugar", "Danlaw Technologies",
    "Dangee Dums", "Datamatics Global", "DCW Limited", "Deccan Gold Mines", "Deep Industries", "Delta Leasing",
    "Den Networks", "Dena Bank", "Dhampur Sugar Mills", "Dhanlaxmi Bank", "Dharamsi Morarji", "Dhanuka Agritech",
    "Dishman Carbogen", "Dixon Technologies", "DLF Limited", "D-Link India", "Donear Industries", "Doshion Water",
    "Dr Lal Pathlabs", "Dredging Corporation", "Dynamatic Technologies", "E.I.D.- Parry (India)", "E-Serve International",
    "East Coast Steel", "Eastern Silk", "Eastern Treads", "Edelweiss Asset", "Educomp Solutions",
    "Eicher Motors", "Elgi Equipments", "Emami Limited", "Emkay Global", "Emporis Projects", "Encore Software",
    "Engineers India", "Entertainment Network", "Equitas Holdings", "Eros International", "Ess Dee Aluminium",
    "Essar Oil", "Essar Shipping", "Essel Propack", "Euro Ceramics", "Euro Multivision", "Everest Kanto",
    "Evergreen Textiles", "Excel Realty N Infra", "Exide Industries", "FACT", "Fairdeal Filaments",
    "FDC Limited", "Federal-Mogul", "Fiberweb India", "Financial Technologies", "Finolex Industries", "Firstsource Solutions"
]

# Find all matching company names in the extracted text
identified_companies = [name for name in company_names if re.search(re.escape(name), text, re.IGNORECASE)]

if not identified_companies:
    identified_companies = ["Unknown Company"]

print(f"Identified company names: {identified_companies}")

# Step 3: Fetch real-time news
def get_news(company_name):
    today = datetime.today().strftime('%Y-%m-%d')
    last_week = (datetime.today() - timedelta(days=3)).strftime('%Y-%m-%d')

    url = f"https://newsapi.org/v2/everything?q={company_name}&from={last_week}&to={today}&sortBy=publishedAt&apiKey=15a07cabb7774accb4d8b32173f9f586"
    response = requests.get(url)
    news_data = response.json()

    if news_data['status'] == 'ok':
        articles = news_data['articles']
        for article in articles:
            article['company'] = company_name
        return articles
    else:
        return []

# Replace 'YOUR_NEWSAPI_KEY' with your actual News API key
all_news_articles = []
for company in identified_companies:
    news_articles = get_news(company)
    all_news_articles.extend(news_articles)

# Step 4: Display the news articles in a table
def display_news(articles):
    news_list = []
    for article in articles:
        news_item = {
            'company': article['company'],
            'title': article['title'],
            'description': article['description'],
            'url': article['url'],
            'publishedAt': article['publishedAt']
        }
        news_list.append(news_item)

    df = pd.DataFrame(news_list)
    return df

news_df = display_news(all_news_articles)

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the new dataset that needs to be labeled
new_data = news_df

# Combine the title and description for sentiment analysis
# Fill any missing values in title or description with an empty string
new_data['title'] = new_data['title'].fillna('')
new_data['description'] = new_data['description'].fillna('')
new_data['text'] = new_data['title'] + ". " + new_data['description']

# Load the pre-trained BERT tokenizer (same one used during training)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess the new data for BERT (same preprocessing as used during training)
def preprocess_data(df):
    return tokenizer(
        df['text'].tolist(),  # Use the combined text (title + description)
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

# Preprocess the new data
new_encodings = preprocess_data(new_data)

# Load the trained model (ensure it's in the same directory as the training output)
model = BertForSequenceClassification.from_pretrained('./results')

# Set model to evaluation mode
model.eval()

# Perform prediction (no labels provided)
with torch.no_grad():
    outputs = model(**new_encodings)
    logits = outputs.logits

# Convert the raw logits to predicted sentiment labels
predicted_labels = torch.argmax(logits, axis=1).numpy()

# Reverse the sentiment mapping to get the original sentiment labels (0 -> negative, 1 -> neutral, 2 -> positive)
reverse_sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
new_data['Predicted_Sentiment'] = [reverse_sentiment_mapping[label] for label in predicted_labels]

# Save the labeled data to a new CSV file
output_file_path = '/content/labeled_financial_data_with_sentiment.csv'
new_data.to_csv(output_file_path, index=False)

output=pd.read_csv('/content/labeled_financial_data_with_sentiment.csv')
news_df = output[output['Predicted_Sentiment'] != 'neutral']

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

expanded_urgency_dictionary=['Accumulate', 'Acquisition', 'After-hours', 'Alert', 'Alliance', 'Allocate', 'Amalgamation', 'Analysis', 'Antitrust', 'Approval', 'Asset', 'Audit', 'Auditor', 'Balance sheet', 'Bankruptcy', 'Bear market', 'Black swan event', 'Breach', 'Bull market', 'Bust', 'Buy', 'Buyback', 'Buyout', 'Cabinet reshuffle', 'Cash flow', 'Caution', 'Cautionary', 'Climb', 'Closing bell', 'Collapse', 'Commission', 'Compliance', 'Concern', 'Consolidation', 'Contraction', 'Correction', 'Cost-cutting', 'Crash', 'Credit crunch', 'Critical', 'Cutoff', 'Deadline', 'Decline', 'Default', 'Deficit', 'Depression', 'Deterioration', 'Diplomacy', 'Diversify', 'Divest', 'Dividend', 'Downgrade', 'Drop', 'Earnings', 'Earnings report', 'Economic crisis', 'Economic decline', 'Election', 'Embargo', 'Emergency', 'Enforcement', 'Equity', 'Escalation', 'Execution', 'Expansion', 'Expiration', 'Expiration date', 'Expiry', 'Failure', 'Fall', 'Fine', 'Fiscal cliff', 'Forecast', 'GDP', 'Geopolitical risk', 'Governance', 'Great_Depression', 'Guidance', 'Hedge', 'High risk', 'Hold', 'Hostile takeover', 'Hyperinflation', 'IPO', 'Immediate', 'Important', 'Inflation', 'Insolvency', 'Integration', 'Interest Rates', 'Invest', 'Investigation', 'Lawsuit', 'Layoffs', 'Legislation', 'Leveraged buyout', 'Liquidate', 'Liquidation', 'Liquidity crisis', 'Loss', 'Mandate', 'Margin', 'Maturity', 'Merger', 'Net income', 'Nosedive', 'O.K.', 'OK', 'Opening bell', 'Operating income', 'Overhead', 'Overnight', 'Penalty', 'Plummet', 'Plunge', 'Policy Change', 'Policy shift', 'Political unrest', 'Portfolio', 'Post-market', 'Pre-market', 'Priority', 'Profit', 'Profit warning', 'Prompt', 'Quarterly results', 'Rally', 'Rebalance', 'Rebound', 'Recession', 'Recommend', 'Red flag', 'Reduce', 'Referendum', 'Regime change', 'Regulation', 'Restructuring', 'Return on investment (ROI)', 'Revenue', 'Revenue miss', 'Risk assessment', 'Risk factor', 'Risk management', 'Ruling', 'Sanction', 'Sanctioned', 'Sanctions', 'Sell', 'Sell-off', 'Settlement', 'Short sell', 'Slowdown', 'Slump', 'Soar', 'Spike', 'Spin-off', 'Stagflation', 'Supply chain disruption', 'Surge', 'Takeover', 'Tariff', 'Time-sensitive', 'Timely', 'Trade War', 'Trading session', 'Unemployment', 'Union', 'Upgrade', 'Urgent', 'Volatility', 'Warn', 'Warning', 'Watchlist', 'Whistleblower', 'Yield', 'abidance', 'acclivity', 'admonish', 'admonitory', 'adorn', 'adulthood', 'alarm', 'alarum', 'alert', 'alerting', 'alignment', 'alinement', 'alive', 'all_right', 'alliance', 'allocate', 'allowance', 'alright', 'amalgamation', 'amass', 'amercement', 'analysis', 'antimonopoly', 'antitrust', 'apportion', 'approve', 'apropos', 'ascent', 'asset', 'audit', 'audited_account', 'augur', 'auspicate', 'authorisation', 'authority', 'authorization', 'awake', 'backlash', 'bankruptcy', 'bargain', 'benefit', 'betoken', 'betray', 'bode', 'bond', 'border', 'bounce', 'bound', 'break', 'break_down', 'break_up', 'bribe', 'brisk', 'burst', 'buy', 'buyback', 'by-product', 'byproduct', 'calculate', 'capital_punishment', 'care', 'carefulness', 'carrying_into_action', 'carrying_out', 'caution', 'cautionary', 'cautiousness', 'cave_in', 'caveat', 'chastening', 'chastisement', 'circumspection', 'circumvent', 'climb', 'climb_up', 'climbing', 'clinical_depression', 'closure', 'clothe', 'coalition', 'collapse', 'collect', 'colonisation', 'colonization', 'colony', 'commission', 'commit', 'compile', 'complaisance', 'compliance', 'compliancy', 'confederation', 'conformation', 'conformity', 'conglomerate', 'consolidation', 'correct', 'correction', 'corrupt', 'counsel', 'counseling', 'counselling', 'count_on', 'countenance', 'crack', 'crack_up', 'crash', 'critical', 'crock_up', 'crumble', 'crumple', 'cumulate', 'deal', 'death', 'death_penalty', 'decay', 'decease', 'decisive', 'declension', 'declination', 'decline', 'declivity', 'deference', 'deficit', 'delicately', 'departure', 'depression', 'depressive_disorder', 'deprivation', 'deprive', 'descent', 'desegregation', 'diminution', 'direction', 'discipline', 'disinvest', 'dive', 'divest', 'dividend', 'do_in', 'dodge', 'downgrade', 'downslope', 'duck', 'due_date', 'earnings', 'economic_crisis', 'elaboration', 'elude', 'empower', 'endorsement', 'endow', 'endue', 'enforcement', 'enlargement', 'enthrone', 'escalation', 'estimate', 'evade', 'evaluate', 'executing', 'execution', 'execution_of_instrument', 'exemplary', 'exit', 'expanding_upon', 'expansion', 'expiration', 'expiry', 'exquisitely', 'failure', 'fall', 'fall_in', 'figure', 'fine', 'finely', 'flop', 'fluctuation', 'forecast', 'foreshadow', 'foretell', 'forethought', 'founder', 'fudge', 'fudge_factor', 'fusion', 'gain', 'gather', 'gift', 'give', 'give_way', 'go_down', 'go_up', 'going', "grease_one's_palms", 'gross_domestic_product', 'gross_profit', 'gross_profit_margin', 'guidance', 'hedge', 'hedge_in', 'hedgerow', 'hedging', 'hoard', 'hunky-dory', 'implementation', 'impression', 'imprimatur', 'imprint', 'indorsement', 'induct', 'indue', 'initial_offering', 'initial_public_offering', 'insolvency', 'inspect', 'instability', 'instruction_execution', 'integrating', 'integration', 'interest', 'invest', 'knock_off', 'lawmaking', 'layoff', 'leeway', 'legislating', 'legislation', 'liquidate', 'liquidation', 'lively', 'loser', 'loss', 'low', 'lucre', 'mandate', 'mandatory', 'margin', 'matureness', 'maturity', 'maturity_date', 'merger', 'merry', 'monish', 'monitory', 'mount', 'mounting', 'mulct', 'murder', 'natural_depression', 'net', 'net_income', 'net_profit', 'neutralise', 'neutralize', 'nonstarter', 'nose_dive', 'nosedive', 'o.k.', 'obligingness', 'ok', 'okay', 'omen', 'ordinance', 'output', 'parry', 'pass_up', 'passing', 'pay_off', 'performance', 'perimeter', 'personnel_casualty', 'pile_up', 'place', 'plus', 'portend', 'portfolio', 'precaution', 'predict', 'prefigure', 'presage', 'produce', 'profit', 'profits', 'prognosis', 'prognosticate', 'prophylactic', 'prostration', 'purchase', 'put', 'put_off', 'qui_vive', 'raise', 'rally', 'rattling', 'rebound', 'reckon', 'recoil', 'rectification', 'red', 'red_ink', 'redemption', 'refuse', 'regularisation', 'regularization', 'regulating', 'regulation', 'reject', 'release', 'repercussion', 'repurchase', 'resile', 'resolution', 'return', 'reverberate', 'ricochet', 'rise', 'roll_up', 'rule', 'sanction', 'scrutinise', 'scrutinize', 'seasonable', 'seasonably', 'seat', 'security_deposit', 'sell', 'settlement', 'shortage', 'shortfall', 'sidestep', 'skirt', 'slaying', 'slump', 'small_town', 'snappy', 'spanking', 'spin-off', 'spring', 'stagflation', 'statute_law', 'steal', 'steering', 'strip', 'submission', 'take_a_hop', 'termination', 'ticket', 'timely', 'tolerance', 'trade', 'tumble', 'turn_a_profit', 'turn_down', 'undress', 'unemployment', 'unification', 'uniting', 'unsuccessful_person', 'upgrade', 'very_well', 'vest', 'village', 'vital', 'volatility', 'wane', 'warning', 'warning_signal', 'warrant', 'waste', 'watchful', 'wax', 'well-timed', 'well_timed', 'worsen', 'writ_of_execution', 'yield', 'zippy']

# Convert urgency words to embeddings
urgency_embeddings = model.encode(expanded_urgency_dictionary)

def check_urgency(word, threshold=0.9):
    # Get the embedding of the target word
    word_embedding = model.encode([word])[0]

    # Calculate cosine similarity with each word in the urgency dictionary
    similarities = util.cos_sim(word_embedding, urgency_embeddings)

    similarities_np = similarities.numpy()

    max_similarity = np.max(similarities_np)

    return max_similarity >= threshold

# Create a list to store the urgent rows
urgent_rows = []

# Iterate over the rows in the news_df DataFrame
for index, row in news_df.iterrows():
    description = row['description']

    # Tokenize the description into words
    words = description.split()

    # Check for urgency in the description
    for word in words:
        if check_urgency(word):
            urgent_rows.append(row)  # Append the entire row if it passes the check
            break

# Create a new DataFrame from the urgent rows
urgent_title_df = pd.DataFrame(urgent_rows)

# Display the new DataFrame with urgent titles
display(urgent_title_df)


