from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from PIL import Image
import pytesseract
import re
from datetime import datetime, timedelta
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import numpy as np
import sqlite3
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


app = Flask(__name__, static_url_path='/static')
CORS(app)


# API keys
COHERE_API_KEY = 'WlAjHTaXW5ElFhvmUNWmcPr97IAiinSuHIqOUOk7'
NEWS_API_KEY = '36d0bf0452d54bdda22c7ecc11caa708'

# Load necessary models
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('./results')
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define urgency dictionary
expanded_urgency_dictionary=['Accumulate', 'Acquisition', 'After-hours', 'Alert', 'Alliance', 'Allocate', 'Amalgamation', 'Analysis', 'Antitrust', 'Approval', 'Asset', 'Audit', 'Auditor', 'Balance sheet', 'Bankruptcy', 'Bear market', 'Black swan event', 'Breach', 'Bull market', 'Bust', 'Buy', 'Buyback', 'Buyout', 'Cabinet reshuffle', 'Cash flow', 'Caution', 'Cautionary', 'Climb', 'Closing bell', 'Collapse', 'Commission', 'Compliance', 'Concern', 'Consolidation', 'Contraction', 'Correction', 'Cost-cutting', 'Crash', 'Credit crunch', 'Critical', 'Cutoff', 'Deadline', 'Decline', 'Default', 'Deficit', 'Depression', 'Deterioration', 'Diplomacy', 'Diversify', 'Divest', 'Dividend', 'Downgrade', 'Drop', 'Earnings', 'Earnings report', 'Economic crisis', 'Economic decline', 'Election', 'Embargo', 'Emergency', 'Enforcement', 'Equity', 'Escalation', 'Execution', 'Expansion', 'Expiration', 'Expiration date', 'Expiry', 'Failure', 'Fall', 'Fine', 'Fiscal cliff', 'Forecast', 'GDP', 'Geopolitical risk', 'Governance', 'Great_Depression', 'Guidance', 'Hedge', 'High risk', 'Hold', 'Hostile takeover', 'Hyperinflation', 'IPO', 'Immediate', 'Important', 'Inflation', 'Insolvency', 'Integration', 'Interest Rates', 'Invest', 'Investigation', 'Lawsuit', 'Layoffs', 'Legislation', 'Leveraged buyout', 'Liquidate', 'Liquidation', 'Liquidity crisis', 'Loss', 'Mandate', 'Margin', 'Maturity', 'Merger', 'Net income', 'Nosedive', 'O.K.', 'OK', 'Opening bell', 'Operating income', 'Overhead', 'Overnight', 'Penalty', 'Plummet', 'Plunge', 'Policy Change', 'Policy shift', 'Political unrest', 'Portfolio', 'Post-market', 'Pre-market', 'Priority', 'Profit', 'Profit warning', 'Prompt', 'Quarterly results', 'Rally', 'Rebalance', 'Rebound', 'Recession', 'Recommend', 'Red flag', 'Reduce', 'Referendum', 'Regime change', 'Regulation', 'Restructuring', 'Return on investment (ROI)', 'Revenue', 'Revenue miss', 'Risk assessment', 'Risk factor', 'Risk management', 'Ruling', 'Sanction', 'Sanctioned', 'Sanctions', 'Sell', 'Sell-off', 'Settlement', 'Short sell', 'Slowdown', 'Slump', 'Soar', 'Spike', 'Spin-off', 'Stagflation', 'Supply chain disruption', 'Surge', 'Takeover', 'Tariff', 'Time-sensitive', 'Timely', 'Trade War', 'Trading session', 'Unemployment', 'Union', 'Upgrade', 'Urgent', 'Volatility', 'Warn', 'Warning', 'Watchlist', 'Whistleblower', 'Yield', 'abidance', 'acclivity', 'admonish', 'admonitory', 'adorn', 'adulthood', 'alarm', 'alarum', 'alert', 'alerting', 'alignment', 'alinement', 'alive', 'all_right', 'alliance', 'allocate', 'allowance', 'alright', 'amalgamation', 'amass', 'amercement', 'analysis', 'antimonopoly', 'antitrust', 'apportion', 'approve', 'apropos', 'ascent', 'asset', 'audit', 'audited_account', 'augur', 'auspicate', 'authorisation', 'authority', 'authorization', 'awake', 'backlash', 'bankruptcy', 'bargain', 'benefit', 'betoken', 'betray', 'bode', 'bond', 'border', 'bounce', 'bound', 'break', 'break_down', 'break_up', 'bribe', 'brisk', 'burst', 'buy', 'buyback', 'by-product', 'byproduct', 'calculate', 'capital_punishment', 'care', 'carefulness', 'carrying_into_action', 'carrying_out', 'caution', 'cautionary', 'cautiousness', 'cave_in', 'caveat', 'chastening', 'chastisement', 'circumspection', 'circumvent', 'climb', 'climb_up', 'climbing', 'clinical_depression', 'closure', 'clothe', 'coalition', 'collapse', 'collect', 'colonisation', 'colonization', 'colony', 'commission', 'commit', 'compile', 'complaisance', 'compliance', 'compliancy', 'confederation', 'conformation', 'conformity', 'conglomerate', 'consolidation', 'correct', 'correction', 'corrupt', 'counsel', 'counseling', 'counselling', 'count_on', 'countenance', 'crack', 'crack_up', 'crash', 'critical', 'crock_up', 'crumble', 'crumple', 'cumulate', 'deal', 'death', 'death_penalty', 'decay', 'decease', 'decisive', 'declension', 'declination', 'decline', 'declivity', 'deference', 'deficit', 'delicately', 'departure', 'depression', 'depressive_disorder', 'deprivation', 'deprive', 'descent', 'desegregation', 'diminution', 'direction', 'discipline', 'disinvest', 'dive', 'divest', 'dividend', 'do_in', 'dodge', 'downgrade', 'downslope', 'duck', 'due_date', 'earnings', 'economic_crisis', 'elaboration', 'elude', 'empower', 'endorsement', 'endow', 'endue', 'enforcement', 'enlargement', 'enthrone', 'escalation', 'estimate', 'evade', 'evaluate', 'executing', 'execution', 'execution_of_instrument', 'exemplary', 'exit', 'expanding_upon', 'expansion', 'expiration', 'expiry', 'exquisitely', 'failure', 'fall', 'fall_in', 'figure', 'fine', 'finely', 'flop', 'fluctuation', 'forecast', 'foreshadow', 'foretell', 'forethought', 'founder', 'fudge', 'fudge_factor', 'fusion', 'gain', 'gather', 'gift', 'give', 'give_way', 'go_down', 'go_up', 'going', "grease_one's_palms", 'gross_domestic_product', 'gross_profit', 'gross_profit_margin', 'guidance', 'hedge', 'hedge_in', 'hedgerow', 'hedging', 'hoard', 'hunky-dory', 'implementation', 'impression', 'imprimatur', 'imprint', 'indorsement', 'induct', 'indue', 'initial_offering', 'initial_public_offering', 'insolvency', 'inspect', 'instability', 'instruction_execution', 'integrating', 'integration', 'interest', 'invest', 'knock_off', 'lawmaking', 'layoff', 'leeway', 'legislating', 'legislation', 'liquidate', 'liquidation', 'lively', 'loser', 'loss', 'low', 'lucre', 'mandate', 'mandatory', 'margin', 'matureness', 'maturity', 'maturity_date', 'merger', 'merry', 'monish', 'monitory', 'mount', 'mounting', 'mulct', 'murder', 'natural_depression', 'net', 'net_income', 'net_profit', 'neutralise', 'neutralize', 'nonstarter', 'nose_dive', 'nosedive', 'o.k.', 'obligingness', 'ok', 'okay', 'omen', 'ordinance', 'output', 'parry', 'pass_up', 'passing', 'pay_off', 'performance', 'perimeter', 'personnel_casualty', 'pile_up', 'place', 'plus', 'portend', 'portfolio', 'precaution', 'predict', 'prefigure', 'presage', 'produce', 'profit', 'profits', 'prognosis', 'prognosticate', 'prophylactic', 'prostration', 'purchase', 'put', 'put_off', 'qui_vive', 'raise', 'rally', 'rattling', 'rebound', 'reckon', 'recoil', 'rectification', 'red', 'red_ink', 'redemption', 'refuse', 'regularisation', 'regularization', 'regulating', 'regulation', 'reject', 'release', 'repercussion', 'repurchase', 'resile', 'resolution', 'return', 'reverberate', 'ricochet', 'rise', 'roll_up', 'rule', 'sanction', 'scrutinise', 'scrutinize', 'seasonable', 'seasonably', 'seat', 'security_deposit', 'sell', 'settlement', 'shortage', 'shortfall', 'sidestep', 'skirt', 'slaying', 'slump', 'small_town', 'snappy', 'spanking', 'spin-off', 'spring', 'stagflation', 'statute_law', 'steal', 'steering', 'strip', 'submission', 'take_a_hop', 'termination', 'ticket', 'timely', 'tolerance', 'trade', 'tumble', 'turn_a_profit', 'turn_down', 'undress', 'unemployment', 'unification', 'uniting', 'unsuccessful_person', 'upgrade', 'very_well', 'vest', 'village', 'vital', 'volatility', 'wane', 'warning', 'warning_signal', 'warrant', 'waste', 'watchful', 'wax', 'well-timed', 'well_timed', 'worsen', 'writ_of_execution', 'yield', 'zippy']
urgency_embeddings = sentence_model.encode(expanded_urgency_dictionary)

# Routes
############################financial literacy chatbot############################################
@app.route('/')
def index():
    return render_template('financial_literacy_chatbot.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method=='GET':
        return render_template('Signup.html')
    if request.method=='POST':
        #Update post code
        return "HELLO"

@app.route('/portfolio')
def portfolio():
    return render_template('stock_news_portfolio.html')

@app.route('/transactions')
def transactions():
    return render_template('transaction_entry.html')

@app.route('/coherenceapi', methods=['POST'])
def coherence_api():
    try:
        data = request.json
        response = requests.post(
            'https://api.cohere.ai/v1/generate',
            json=data,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {COHERE_API_KEY}'
            }
        )
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        error_message = e.response.json() if e.response else str(e)
        print("Error from Coherence API:", error_message)
        return jsonify({
            'message': 'Failed to fetch data from Coherence API',
            'error': error_message
        }), 500

########################stock news portfolio######################
@app.route('/processPortfolioImage', methods=['POST'])
def process_portfolio_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    image = Image.open(file)
    text = pytesseract.image_to_string(image)

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
    "FDC Limited", "Federal-Mogul", "Fiberweb India", "Financial Technologies", "Finolex Industries", "Firstsource Solutions"]
    # Identify companies mentioned in the extracted text
    identified_companies = [name for name in company_names if re.search(re.escape(name), text, re.IGNORECASE)]
    if not identified_companies:
        identified_companies = ["Unknown Company"]

    # Step 1: Fetch news articles for the identified companies for the last 3 days
    all_news_articles = []
    for company in identified_companies:
        articles = get_news(company)
        for article in articles:
            article['company'] = company
        all_news_articles.extend(articles)

    news_df = pd.DataFrame(all_news_articles)
    if news_df.empty:
        return jsonify({"result": []})

    # Step 2: Perform sentiment analysis and filter out 'neutral' news
    news_df['title'] = news_df['title'].fillna('')
    news_df['description'] = news_df['description'].fillna('')
    news_df['text'] = news_df['title'] + ". " + news_df['description']

    inputs = bert_tokenizer(news_df['text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(**inputs)
        predictions = torch.argmax(outputs.logits, axis=1).numpy()

    # Define sentiment mapping and filter only positive and negative news
    reverse_sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    news_df['Predicted_Sentiment'] = [reverse_sentiment_mapping[pred] for pred in predictions]
    filtered_news_df = news_df[news_df['Predicted_Sentiment'].isin(['positive', 'negative'])]

    # Step 3: Check urgency for filtered news and apply the 0.9 threshold
    urgent_news = []
    for _, row in filtered_news_df.iterrows():
        if any(check_urgency(word) for word in row['description'].split()):
            urgent_news.append(row)

    urgent_df = pd.DataFrame(urgent_news)
    result = urgent_df[['company', 'title', 'description', 'Predicted_Sentiment']].to_dict(orient='records')

    return jsonify({"result": result})

# Function to get news articles for the last 3 days
def get_news(company_name):
    today = datetime.today().strftime('%Y-%m-%d')
    last_week = (datetime.today() - timedelta(days=3)).strftime('%Y-%m-%d')
    url = f"https://newsapi.org/v2/everything?q={company_name}&from={last_week}&to={today}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    return response.json().get('articles', [])

# Urgency check function with a similarity threshold of 0.9
def check_urgency(word, threshold=0.9):
    word_embedding = sentence_model.encode([word])[0]
    similarities = util.cos_sim(word_embedding, urgency_embeddings)
    return np.max(similarities.numpy()) >= threshold

####################transaction entry#####################################


hypernym_categories = {
    'food': [
        wn.synset('food.n.01'), wn.synset('meal.n.01'), wn.synset('snack.n.01'), wn.synset('drink.n.01'),        wn.synset('grocery.n.01'), wn.synset('produce.n.01'), wn.synset('beverage.n.01'), wn.synset('cooking.n.01'),
        wn.synset('eating.n.01'), wn.synset('cuisine.n.01'), wn.synset('dairy_product.n.01'), wn.synset('meat.n.01'),
        wn.synset('seafood.n.01'), wn.synset('spice.n.01'), wn.synset('ingredient.n.01')
    ],
    'social_life': [
        wn.synset('recreation.n.01'), wn.synset('celebration.n.01'), wn.synset('outing.n.01'), wn.synset('party.n.01'),
        wn.synset('dancing.n.01'), wn.synset('social_event.n.01'), wn.synset('entertainment.n.01'), wn.synset('concert.n.01'),
        wn.synset('movie.n.01'), wn.synset('gathering.n.01'), wn.synset('festival.n.01'), wn.synset('show.n.01'),
        wn.synset('nightclub.n.01'), wn.synset('bar.n.01'), wn.synset('pub.n.01'), wn.synset('celebration.n.01')
    ],
    'transportation': [
        wn.synset('vehicle.n.01'), wn.synset('transportation.n.01'), wn.synset('public_transport.n.01'), wn.synset('taxi.n.01'),
        wn.synset('airplane.n.01'), wn.synset('car.n.01'), wn.synset('train.n.01'), wn.synset('bus.n.01'),
        wn.synset('fare.n.01'), wn.synset('subway.n.01'), wn.synset('railway.n.01'), wn.synset('transport.n.01'),
        wn.synset('commute.n.01'), wn.synset('travel.n.01'), wn.synset('motor_vehicle.n.01'), wn.synset('bicycle.n.01'),
        wn.synset('ticket.n.01'), wn.synset('road.n.01')
    ],
    'entertainment': [
        wn.synset('culture.n.01'), wn.synset('art.n.01'), wn.synset('entertainment.n.01'), wn.synset('concert.n.01'),
        wn.synset('museum.n.01'), wn.synset('movie.n.01'), wn.synset('game.n.01'), wn.synset('sport.n.01'),
        wn.synset('amusement.n.01'), wn.synset('theater.n.01'), wn.synset('exhibition.n.01'), wn.synset('show.n.01'),
        wn.synset('play.n.01'), wn.synset('performance.n.01'), wn.synset('event.n.01'), wn.synset('hobby.n.01')
    ],
    'household': [
        wn.synset('household.n.01'), wn.synset('furniture.n.01'), wn.synset('appliance.n.01'), wn.synset('utility.n.01'),
        wn.synset('cleaning.n.01'), wn.synset('kitchenware.n.01'), wn.synset('decoration.n.01'), wn.synset('bedding.n.01'),
        wn.synset('laundry.n.01'), wn.synset('repair.n.01'), wn.synset('gardening.n.01'), wn.synset('maintenance.n.01'),
        wn.synset('utensil.n.01'), wn.synset('fixture.n.01'), wn.synset('housework.n.01')
    ],
    'shopping': [
        wn.synset('clothing.n.01'), wn.synset('footwear.n.01'), wn.synset('accessory.n.01'), wn.synset('outerwear.n.01'),
        wn.synset('toiletry.n.01'), wn.synset('cosmetic.n.01'), wn.synset('jewelry.n.01'), wn.synset('apparel.n.01'),
        wn.synset('bag.n.01'), wn.synset('watch.n.01'), wn.synset('fashion.n.01'), wn.synset('retail.n.01'),
        wn.synset('boutique.n.01'), wn.synset('department_store.n.01'), wn.synset('shopping.n.01')
    ],
    'health': [
        wn.synset('health.n.01'), wn.synset('medicine.n.01'), wn.synset('therapy.n.01'), wn.synset('fitness.n.01'),
        wn.synset('exercise.n.01'), wn.synset('training.n.01'), wn.synset('meditation.n.01'), wn.synset('nutrition.n.01'),
        wn.synset('doctor.n.01'), wn.synset('hospital.n.01'), wn.synset('wellness.n.01'), wn.synset('pharmacy.n.01'),
        wn.synset('clinic.n.01'), wn.synset('dentist.n.01'), wn.synset('treatment.n.01'), wn.synset('nurse.n.01'),
        wn.synset('diet.n.01')
    ],
    'education': [
        wn.synset('education.n.01'), wn.synset('schooling.n.01'), wn.synset('book.n.01'), wn.synset('course.n.01'),
        wn.synset('lecture.n.01'), wn.synset('research.n.01'), wn.synset('workshop.n.01'), wn.synset('library.n.01'),
        wn.synset('stationery.n.01'), wn.synset('tuition.n.01'), wn.synset('student.n.01'), wn.synset('class.n.01'),
        wn.synset('teacher.n.01'), wn.synset('exam.n.01'), wn.synset('assignment.n.01'), wn.synset('university.n.01')
    ],
    'gift': [
        wn.synset('gift.n.01'), wn.synset('present.n.01'), wn.synset('souvenir.n.01'), wn.synset('donation.n.01'),
        wn.synset('offering.n.01'), wn.synset('keepsake.n.01'), wn.synset('award.n.01'), wn.synset('memento.n.01'),
        wn.synset('prize.n.01'), wn.synset('trophy.n.01'), wn.synset('celebration.n.01'), wn.synset('token.n.01')
    ],
    'others': [
        wn.synset('artifact.n.01'), wn.synset('object.n.01'), wn.synset('thing.n.01'), wn.synset('item.n.01'),
        wn.synset('material.n.01'), wn.synset('equipment.n.01'), wn.synset('supply.n.01'), wn.synset('instrumentality.n.01'),
        wn.synset('device.n.01'), wn.synset('commodity.n.01'), wn.synset('resource.n.01'), wn.synset('tool.n.01')
    ]
}

# Flatten synsets for each category to enable direct comparison
flattened_hypernym_categories = {cat: synsets for cat, synsets in hypernym_categories.items()}

lemmatizer = WordNetLemmatizer()

direct_mappings = {
    'banana': 'food', 'starbucks': 'food', 'electricity bill': 'household', 'fare': 'transportation', 
    'ticket': 'entertainment', 'cafe': 'food', 'shopping': 'shopping', 'phone': 'entertainment', 
    'rupee book': 'education', 'water': 'food', 'groceries': 'food', 'movie ticket': 'entertainment',
    'bus fare': 'transportation', 'doctor visit': 'health', 'furniture': 'household', 'books': 'education', 
    'electricity': 'household', 'concert': 'social_life', 'taxi ride': 'transportation', 'rent':'household', 'mango':'food',
    'apple': 'food'
}

# Helper function to check if any synset in a hypernym path matches the target synsets for a category
def is_hypernym_in_path(synset, target_hypernyms):
    hypernym_paths = synset.hypernym_paths()  # Get all hypernym paths
    for path in hypernym_paths:
        for ancestor in path:
            if ancestor in target_hypernyms:
                return True
    return False

# Get category by hypernym path and semantic similarity
def get_category(item):
    # Take only the first word if the item has multiple words
    first_word = item.split()[0]
    
    # Normalize and lemmatize the first word
    lemmatized_item = lemmatizer.lemmatize(first_word.lower())
    
    # Check direct mappings first
    if lemmatized_item in direct_mappings:
        return direct_mappings[lemmatized_item]
    
    # Get synsets for the item as a noun
    item_synsets = wn.synsets(lemmatized_item, pos=wn.NOUN)
    for synset in item_synsets:
        # Check if any synset in the item's hypernym path matches a category
        for category, target_hypernyms in hypernym_categories.items():
            if is_hypernym_in_path(synset, target_hypernyms):
                return category

    # As a last resort, use semantic similarity to find the closest category
    best_category = 'others'
    max_similarity = 0
    for synset in item_synsets:
        for category, target_hypernyms in hypernym_categories.items():
            for target_synset in target_hypernyms:
                similarity = synset.path_similarity(target_synset)
                if similarity and similarity > max_similarity:
                    max_similarity = similarity
                    best_category = category
    return best_category

# Initialize database connection
def get_db_connection():
    conn = sqlite3.connect("categorised_transaction.db")
    conn.row_factory = sqlite3.Row
    return conn

# Add expense entry
@app.route("/add_expense", methods=["POST"])
def add_expense():
    data = request.json
    description = data['description']
    amount = data['amount']
    importance = data['importance']
    timestamp = datetime.now()
    category = get_category(description)
    type_ = "Expense"
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO transactions (amount, item, category, Timestamp, type, importance) VALUES (?, ?, ?, ?, ?, ?)",
        (amount, description, category, timestamp, type_, importance)
    )
    conn.commit()
    conn.close()
    return jsonify(success=True)

# Add income entry
@app.route("/add_income", methods=["POST"])
def add_income():
    data = request.json
    description = data['description']
    amount = data['amount']
    category = data['category']
    importance = data['importance']
    timestamp = datetime.now()
    type_ = "Income"
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO transactions (amount, item, category, Timestamp, type, importance) VALUES (?, ?, ?, ?, ?, ?)",
        (amount, description, category, timestamp, type_, importance)
    )
    conn.commit()
    conn.close()
    return jsonify(success=True)

# Get all transactions
@app.route("/get_transactions", methods=["GET"])
def get_transactions():
    conn = get_db_connection()
    transactions = conn.execute("SELECT * FROM transactions ORDER BY Timestamp DESC").fetchall()
    conn.close()
    return jsonify([dict(row) for row in transactions])

@app.route("/voice_record_expense", methods=["POST"])
def voice_record_expense():
    import pyaudio
    import numpy as np
    import speech_recognition as sr
    import io
    from datetime import datetime
    import time
    import spacy
    import nltk
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')


    nlp = spacy.load("en_core_web_sm")

    RATE = 16000  # Sample rate
    CHUNK = 1024  # Number of frames per buffer

    def is_silent(data, threshold=500):
        """Returns 'True' if below the 'silent' threshold"""
        return np.abs(np.frombuffer(data, dtype=np.int16)).max() < threshold

    def record_audio():
        recognizer = sr.Recognizer()
        
        audio_data = io.BytesIO()
        p = pyaudio.PyAudio()
        
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Listening for expense entry...")
        silence_start = None
        audio_chunks = []
        
        while True:
            data = stream.read(CHUNK)
            audio_chunks.append(data)
            
            if is_silent(data):
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > 3:  # Stop recording after 3 seconds of silence
                    break
            else:
                silence_start = None
        
        print("Recording complete")
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        audio_data.write(b''.join(audio_chunks))
        audio_data.seek(0)
        return audio_data, recognizer

    def speech_to_text(audio_data, recognizer):
        audio = sr.AudioData(audio_data.read(), RATE, 2)
        try:
            text = recognizer.recognize_google(audio)
            print("Recognized Text:", text)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def extract_amount_item(transaction):
        # Tokenize the transaction text
        tokens = nltk.word_tokenize(transaction)
        
        # POS tagging
        pos_tags = nltk.pos_tag(tokens)
        
        # Compile a list of common currency words
        currency_terms = {"rs", "rupees", "bucks", "dollars", "pounds", "cost", "rupee", "lakh", "crore", "million"}
        
        # Initialize amount as "Unknown" and a flag for currency presence
        amount = "Unknown"
        currency_found = False
        
        # Find the closest number to the currency term if present
        for i, token in enumerate(tokens):
            if token.lower() in currency_terms:
                currency_found = True
                # Check for a number before or after the currency term
                if i > 0 and re.match(r'\d+(\.\d+)?', tokens[i-1]):  # Check previous token
                    amount = tokens[i-1]
                elif i < len(tokens) - 1 and re.match(r'\d+(\.\d+)?', tokens[i+1]):  # Check next token
                    amount = tokens[i+1]
                break  # Stop after finding the first currency term
    
        # If no currency term found, look for any number in the sentence
        if not currency_found:
            for token in tokens:
                if re.match(r'\d+(\.\d+)?', token):  # Find any number
                    amount = token
                    break  # Stop after finding the first number
    
        # Extract nouns as potential items, filtering out currency terms
        item_tokens = []
        for word, tag in pos_tags:
            # Collect nouns (NN, NNS) that are not currency terms
            if tag in ['NN', 'NNS'] and word.lower() not in currency_terms:
                item_tokens.append(word)
        
        # Join tokens to form the item name, if available
        item = " ".join(item_tokens) if item_tokens else "Unknown"
        
        return amount.strip(), item.strip()

    # Record and transcribe the audio
    audio_data, recognizer = record_audio()
    text = speech_to_text(audio_data, recognizer)
    
    if not text:
        return jsonify({"error": "Could not understand audio"}), 400

    # Extract amount and item from the recognized text
    amt, item = extract_amount_item(text)
    
    if amt == 0 or not item:
        return jsonify({"error": "Could not extract amount or item from the text"}), 400

    # Determine the category based on the item
    category = get_category(item)
    timestamp = datetime.now()
    type_ = 'Expense'
    importance = 'Not Important'  # Default; you can change this logic as needed

    # Insert the transaction into the database
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO transactions (amount, item, category, Timestamp, type, importance) VALUES (?, ?, ?, ?, ?, ?)",
        (amt, item, category, timestamp, type_, importance)
    )
    conn.commit()
    conn.close()

    return jsonify(success=True)

####################personal dashboard##########################

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import sqlite3

# Initialize Dash application inside Flask
dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname='/dashboard/'
)

# Define the layout of the Dash app with navbar
dash_app.layout = html.Div(
    style={'backgroundColor': '#0d1117', 'color': '#c9d1d9', 'padding': '2em'},
    children=[
        # Navbar
        html.Div(
            style={
                'backgroundColor': '#161b22',
                'padding': '1em',
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'position': 'sticky',
                'top': 0,
                'zIndex': 1000,
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.5)'
            },
            children=[
                html.Div(
                    style={'display': 'flex', 'alignItems': 'center'},
                    children=[
                        html.Img(src='../static/kubera.png', style={'height': '45px', 'width': '45px', 'border': 'solid 2px black', 'borderRadius': '10px', 'marginRight': '0.5em'}),
                        html.A("Kubera Financial Manager", href="/", style={'color': '#c9d1d9', 'textDecoration': 'none', 'fontWeight': '700'})
                    ]
                ),
                html.Div(
                    style={'display': 'flex', 'gap': '1em'},
                    children=[
                        html.A("Transactions", href="/transactions", style={'color': '#c9d1d9', 'textDecoration': 'none', 'fontWeight': '500'}),
                        html.A("Dashboard", href="/dashboard", style={'color': '#c9d1d9', 'textDecoration': 'none', 'fontWeight': '500'}),
                        html.A("Stock News Portfolio", href="/portfolio", style={'color': '#c9d1d9', 'textDecoration': 'none', 'fontWeight': '500'}),
                        html.A("Finbot", href="/finbot", style={'color': '#c9d1d9', 'textDecoration': 'none', 'fontWeight': '500'}),
                        html.A("Sign Up", href="/signup", style={'color': '#c9d1d9', 'textDecoration': 'none', 'fontWeight': '500'})
                    ]
                )
            ]
        ),

        # Page Title
        html.H1("Personal Financial Dashboard", style={'textAlign': 'center', 'color': '#58a6ff', 'marginTop': '1em'}),

        # Filter section for Time Range and Transaction Type
        html.Div([
            html.Label("Select Time Range:", style={'fontSize': '1.2em', 'marginRight': '1em'}),
            dcc.Dropdown(
                id='time-range',
                options=[
                    {'label': 'Daily', 'value': 'Daily'},
                    {'label': 'Weekly', 'value': 'Weekly'},
                    {'label': 'Monthly', 'value': 'Monthly'},
                    {'label': 'Yearly', 'value': 'Yearly'},
                    {'label': 'Custom Dates', 'value': 'Custom'}
                ],
                value='Monthly',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'}
            ),
            dcc.DatePickerRange(
                id='custom-date-picker',
                start_date=None,
                end_date=None,
                style={'display': 'none', 'marginLeft': '1em', 'verticalAlign': 'middle'}
            ),
            html.Label("Transaction Type:", style={'fontSize': '1.2em', 'marginLeft': '2em', 'marginRight': '1em'}),
            dcc.Dropdown(
                id='transaction-type',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Income', 'value': 'Income'},
                    {'label': 'Expense', 'value': 'Expense'}
                ],
                value='All',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'}
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '2em'}),

        # Graphs section with 2-column layout
        html.Div([
            html.Div([
                dcc.Graph(id='transactions-time-series', config={'displayModeBar': False}),
            ], style={'width': '100%', 'marginBottom': '2em'}),

            html.Div([
                dcc.Graph(id='category-pie-chart', config={'displayModeBar': False}),
                dcc.Graph(id='monthly-spending-bar-chart', config={'displayModeBar': False}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'gap': '2em'}),
        ]),
    ]
)

# Callback to update the date picker range based on the selected data
@dash_app.callback(
    Output('custom-date-picker', 'style'),
    Output('custom-date-picker', 'start_date'),
    Output('custom-date-picker', 'end_date'),
    Input('time-range', 'value')
)
def show_date_picker(time_range):
    df_cleaned = get_cleaned_data()
    min_date = df_cleaned['Date'].min()
    max_date = df_cleaned['Date'].max()
    if time_range == 'Custom':
        return {'display': 'block', 'marginLeft': '1em'}, min_date, max_date
    return {'display': 'none'}, min_date, max_date

# Callback to update graphs based on date range and transaction type
@dash_app.callback(
    [Output('transactions-time-series', 'figure'),
     Output('category-pie-chart', 'figure'),
     Output('monthly-spending-bar-chart', 'figure')],
    [Input('time-range', 'value'),
     Input('transaction-type', 'value'),
     Input('custom-date-picker', 'start_date'),
     Input('custom-date-picker', 'end_date')]
)
def update_graphs(time_range, transaction_type, start_date, end_date):
    df_cleaned = get_cleaned_data()

    # Filter data based on transaction type
    filtered_df = df_cleaned if transaction_type == 'All' else df_cleaned[df_cleaned['Type'] == transaction_type]

    # Filter data based on selected date range
    if time_range == 'Custom':
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(start_date)) & (filtered_df['Date'] <= pd.to_datetime(end_date))]
    elif time_range == 'Daily':
        filtered_df = filtered_df.groupby(filtered_df['Date'].dt.date).agg({'Amount': 'sum'}).reset_index()
    elif time_range == 'Weekly':
        filtered_df = filtered_df.groupby(filtered_df['Date'].dt.to_period('W').apply(lambda r: r.start_time)).agg({'Amount': 'sum'}).reset_index()
    elif time_range == 'Monthly':
        filtered_df = filtered_df.groupby(filtered_df['Date'].dt.to_period('M').apply(lambda r: r.start_time)).agg({'Amount': 'sum'}).reset_index()
    elif time_range == 'Yearly':
        filtered_df = filtered_df.groupby(filtered_df['Date'].dt.to_period('Y').apply(lambda r: r.start_time)).agg({'Amount': 'sum'}).reset_index()

    # Ensure data is available for plotting
    if filtered_df.empty:
        return {}, {}, {}

    # Transactions over time line chart
    time_series_fig = px.line(
        filtered_df,
        x='Date',
        y='Amount',
        title='Transactions Over Time',
        template='plotly_dark'
    )
    time_series_fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor='#21262d', plot_bgcolor='#21262d')
    
    # Category-wise spending pie chart
    pie_df = df_cleaned[(df_cleaned['Date'] >= pd.to_datetime(start_date)) & (df_cleaned['Date'] <= pd.to_datetime(end_date))] if time_range == 'Custom' else df_cleaned[df_cleaned['Date'].isin(filtered_df['Date'])]
    pie_df = pie_df[pie_df['Type'] == transaction_type] if transaction_type != 'All' else pie_df
    category_pie_chart = px.pie(
        pie_df,
        names='Category',
        values='Amount',
        title='Category-wise Distribution',
        template='plotly_dark'
    )
    category_pie_chart.update_layout(margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor='#21262d')

    # Monthly spending bar chart
    filtered_df['Month'] = filtered_df['Date'].dt.to_period('M').astype(str)
    monthly_spending_fig = px.bar(
        filtered_df,
        x='Month',
        y='Amount',
        title='Monthly Spending',
        template='plotly_dark'
    )
    monthly_spending_fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor='#21262d', plot_bgcolor='#21262d')

    return time_series_fig, category_pie_chart, monthly_spending_fig

def get_cleaned_data():
    conn = sqlite3.connect("categorised_transaction.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM transactions;")
    df = pd.DataFrame(cur.fetchall())
    df.columns = ['transaction_id', 'Date', 'Category', 'Description', 'Amount', 'Type', 'Importance']
    df.drop('transaction_id', axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    conn.close()
    return df

####################Query system########################

api_endpoint = "https://api.cohere.ai/v1/generate"

@app.route('/query_system')
def query_to_sql_page():
    return render_template('query_system.html')

@app.route('/generate-sql', methods=['POST'])
def generate_sql():
    data = request.json
    question = data.get('question', '')

    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Convert the following question into an SQLite query based on the 'transactions' table structure.

    Table: transactions
    Columns:
      - transaction_id (INTEGER, Primary Key)
      - Timestamp (DATETIME)
      - category (VARCHAR(30))
      - item (VARCHAR(50))
      - amount (INT)
      - type (VARCHAR(10))
      - importance (VARCHAR(20))

    The category column has the following predefined values:
    [food, social_life, transportation, entertainment, household, shopping, health, education, gift, others]

    The type column has the following predefined values:
    [Expense, Income]

    The importance column has the following predefined values:
    [Important, Not Important]

    For specific items please use the item column to filter.

    Question: {question}

    SQL Query:
    """


    payload = {
        "model": "command-xlarge-nightly",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.3,
        "k": 0,
        "p": 0.75,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop_sequences": [";"]
    }

    try:
        import logging
        # Send the request to Cohere API
        response = requests.post(api_endpoint, json=payload, headers=headers)
        response.raise_for_status()

        response_data = response.json()
        if 'generations' not in response_data or not response_data['generations']:
            return jsonify(error="Cohere API did not return a valid SQL query."), 500

        # Extract the SQL query text
        sql_statement = response_data['generations'][0]['text'].strip()
        logging.debug(f"Generated SQL (raw): {sql_statement}")
        

        # Remove any markdown or code block formatting from the SQL query
        sql_statement = re.sub(r"```[a-z]*\n*", "", sql_statement).replace("```", "").strip()
        logging.debug(f"Generated SQL (cleaned): {sql_statement}")
        print(sql_statement)

        # Execute the SQL query on the transactions database
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql_statement.strip())
            rows = cursor.fetchall()
            conn.close()

            # Check if rows are empty
            if not rows:
                return jsonify(result="No data found for the query.")

            # Process results depending on the structure
            if len(rows) == 1 and len(rows[0]) == 1:
                # Single value result
                return jsonify(result=rows[0][0])
            elif len(rows) == 1:
                # Single row with multiple columns
                columns = [description[0] for description in cursor.description]
                result = dict(zip(columns, rows[0]))
                return jsonify(result=result)
            else:
                # Multiple rows
                columns = [description[0] for description in cursor.description]
                results = [dict(zip(columns, row)) for row in rows]
                return jsonify(result=results)

        except sqlite3.Error as e:
            conn.close()
            error_message = f"SQL execution error: {str(e)}"
            print(error_message)
            return jsonify(error=error_message), 500

    except requests.exceptions.RequestException as e:
        error_message = e.response.json() if e.response else str(e)
        logging.error(f"Request to Cohere API failed: {error_message}")
        return jsonify(error="Request failed: " + error_message), 500
    except ValueError:
        logging.error("Unexpected response from Cohere API.")
        return jsonify(error="Unexpected response from Cohere API."), 500

#############################goal setting########################################

@app.route('/set-goal-page')
def set_goal_page():
    return render_template('goal_setting.html')

# Route for Goal Setting
@app.route('/set-goal', methods=['POST'])
def set_goal():
    try:
        # Parse user input from the request
        data = request.json
        target_amount = float(data['targetAmount'])
        target_period = int(data['targetPeriod'])

        if not isinstance(target_amount, (int, float)):
            return jsonify({"error": "targetAmount must be a number"}), 400
        if not isinstance(target_period, int):
            return jsonify({"error": "targetPeriod must be an integer"}), 400
        
        # Fetch data from database
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM transactions WHERE Importance = 'Not Important' AND Type = 'Expense'", conn)
        conn.close()

        # Data pre-processing and binning logic
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])


        
        # Check if columns are in lowercase or uppercase
        df = df[(df['importance'] == 'Not Important') & (df['type'] == 'Expense')]

        df = df.sort_values('Timestamp')
        base_date = df['Timestamp'].min()

        print("Debug: Target period as int:", target_period)
        print("Debug: First few rows of df:", df.head())

        
        # Calculate the number of months since the base_date
        df['Months_Since_Base'] = (df['Timestamp'].dt.year - base_date.year) * 12 + (df['Timestamp'].dt.month - base_date.month)
        df['Time_Bin'] = (df['Months_Since_Base'] // target_period).astype(int)

        # Calculate the start and end date for each bin using apply
        df['Bin_Start'] = df['Time_Bin'].apply(lambda x: base_date + pd.DateOffset(months=x * target_period))

        # Ensure Bin_Start is a datetime object
        df['Bin_Start'] = pd.to_datetime(df['Bin_Start'], errors='coerce')

        # Debugging output to check if conversion was successful
        print("Bin_Start Data Types:")
        print(df['Bin_Start'].dtypes)

        # Check if there are any NaT values in Bin_Start
        if df['Bin_Start'].isnull().any():
            print("Warning: Some Bin_Start values could not be converted to datetime.")

        # Calculate Bin_End
        df['Bin_End'] = df['Bin_Start'] + pd.DateOffset(months=target_period) - pd.DateOffset(days=1)

        # Ensure Bin_End is a datetime object
        df['Bin_End'] = pd.to_datetime(df['Bin_End'], errors='coerce')

        # After calculating Bin_End
        print("Bin_Start values:")
        print(df['Bin_Start'])

        print("Bin_End values:")
        print(df['Bin_End'])

        # Check the types of the columns involved in the next operation
        print("Data types of columns involved in calculations:")
        print(df.dtypes)

        # Debugging output to check if conversion was successful
        print("Bin_End Data Types:")
        print(df['Bin_End'].dtypes)

        # Check if there are any NaT values in Bin_End
        if df['Bin_End'].isnull().any():
            print("Warning: Some Bin_End values could not be calculated.")

        print("\nDataset after filtering for non-important expenses and applying time bins with start and end dates:")
        print(df[['Timestamp', 'amount', 'category', 'Time_Bin', 'Bin_Start', 'Bin_End']].head())


        # Calculate category-wise spending summary
        bin_category_summary = df.groupby(['Time_Bin', 'category', 'Bin_Start', 'Bin_End']).agg(
            Total_Spent=('amount', 'sum'),
            Average_Spent=('amount', 'mean')
        ).reset_index()

        print("\nCategory-Wise Spending Summary per Time Bin (with Start and End Dates):")
        print(bin_category_summary)

        # Define category weights
        category_weights = {
            'food': 0.2,
            'social_life': 0.7,
            'transportation': 0.3,
            'entertainment': 0.6,
            'household': 0.4,
            'shopping': 0.5,
            'health': 0.3,
            'education': 0.2,
            'gift': 0.5,
            'others': 0.4
        }

        # Calculate the target savings per bin
        target_savings_per_bin = target_amount / (len(bin_category_summary['Time_Bin'].unique()))

        # For each time bin, calculate how much to save in each category
        bin_saving_suggestions = []

        for bin_id, bin_data in bin_category_summary.groupby('Time_Bin'):
            bin_savings = {}
            # Calculate total weighted spending only for non-zero average spending values
            total_weighted_spending = sum(
                row['Average_Spent'] * category_weights.get(row['category'], 0)
                for _, row in bin_data.iterrows() if row['Average_Spent'] > 0
            )

            if total_weighted_spending > 0:
                # Calculate target savings for each category in this bin
                for _, row in bin_data.iterrows():
                    category = row['category']
                    avg_spent = row['Average_Spent']
                    weight = category_weights.get(category, 0)

                    # Proportionate savings based on weight and average spending
                    category_target_saving = (weight * avg_spent / total_weighted_spending) * target_savings_per_bin
                    bin_savings[category] = category_target_saving
            else:
                # If no spending in this bin, set all savings suggestions to zero
                for _, row in bin_data.iterrows():
                    bin_savings[row['category']] = 0.0

            bin_saving_suggestions.append((bin_id, bin_savings))

        # Display bin saving suggestions
        for bin_id, suggestions in bin_saving_suggestions:
            print(f"\nSuggestions for Bin {bin_id}:")
            for category, saving in suggestions.items():
                print(f"- Save {saving:.2f} from {category}")

        from collections import defaultdict

        # Initialize a dictionary to store cumulative savings for each category
        cumulative_savings = defaultdict(float)
        bin_count = len(bin_saving_suggestions)

        # Sum up savings suggestions for each category across all bins
        for _, suggestions in bin_saving_suggestions:
            for category, saving in suggestions.items():
                cumulative_savings[category] += saving

        # Calculate the average savings recommendation per category
        average_savings_recommendations = {category: str(float(cumulative_savings[category])) for category in cumulative_savings}
        
        # Prepare recommendations
        print("Debug: Final saving recommendations:", average_savings_recommendations)
        return jsonify(savingRecommendations=average_savings_recommendations)

    except Exception as e:
        print("Error in '/set-goal':", str(e))
        return jsonify(error="Internal Server Error: " + str(e)), 500



# Run the app
if __name__ == "__main__":
    app.run(port=3000, debug=True)


