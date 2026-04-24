from dotenv import load_dotenv
import os

load_dotenv()
import sys
import pickle
import base64
import numpy as np
import requests
from flask import Flask, request, render_template, jsonify
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from shc_engine import scrape_shc_village_data

if np.__version__.startswith("1."):
    sys.modules["numpy._core"] = np

app = Flask(__name__)

# ── API Keys ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL        = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

try:
    model = pickle.load(open("xgb_model.pkl", "rb"))
    le    = pickle.load(open("encoder.pkl",   "rb"))
    print("✅ ML models loaded successfully.")
    print(f"   Classes: {list(le.classes_)}")
except Exception as e:
    print(f"CRITICAL ERROR: ML models could not be loaded: {e}")

# ─────────────────────────────────────────────
#  INDIAN SOIL DATABASE  (ICAR-based N, P, K, pH)
# ─────────────────────────────────────────────
SOIL_DB = {
    'hyderabad':    {'N': 60, 'P': 40, 'K': 40, 'ph': 7.2},
    'warangal':     {'N': 55, 'P': 35, 'K': 38, 'ph': 7.5},
    'nizamabad':    {'N': 65, 'P': 42, 'K': 45, 'ph': 7.0},
    'karimnagar':   {'N': 58, 'P': 38, 'K': 40, 'ph': 7.3},
    'khammam':      {'N': 62, 'P': 40, 'K': 42, 'ph': 6.8},
    'nalgonda':     {'N': 50, 'P': 30, 'K': 35, 'ph': 7.8},
    'mahbubnagar':  {'N': 52, 'P': 32, 'K': 36, 'ph': 7.6},
    'adilabad':     {'N': 58, 'P': 38, 'K': 40, 'ph': 7.0},
    'medak':        {'N': 55, 'P': 36, 'K': 38, 'ph': 7.2},
    'vijayawada':   {'N': 68, 'P': 45, 'K': 48, 'ph': 6.9},
    'visakhapatnam':{'N': 65, 'P': 42, 'K': 45, 'ph': 6.7},
    'guntur':       {'N': 70, 'P': 48, 'K': 50, 'ph': 6.8},
    'kurnool':      {'N': 55, 'P': 35, 'K': 38, 'ph': 7.6},
    'tirupati':     {'N': 52, 'P': 33, 'K': 36, 'ph': 7.1},
    'rajahmundry':  {'N': 65, 'P': 44, 'K': 46, 'ph': 6.8},
    'nellore':      {'N': 62, 'P': 40, 'K': 44, 'ph': 7.0},
    'kadapa':       {'N': 55, 'P': 36, 'K': 38, 'ph': 7.4},
    'anantapur':    {'N': 48, 'P': 30, 'K': 33, 'ph': 7.8},
    'chittoor':     {'N': 55, 'P': 36, 'K': 40, 'ph': 7.0},
    'bangalore':    {'N': 55, 'P': 45, 'K': 35, 'ph': 6.5},
    'bengaluru':    {'N': 55, 'P': 45, 'K': 35, 'ph': 6.5},
    'mysore':       {'N': 60, 'P': 42, 'K': 38, 'ph': 6.3},
    'hubli':        {'N': 58, 'P': 40, 'K': 40, 'ph': 6.8},
    'mangalore':    {'N': 70, 'P': 50, 'K': 55, 'ph': 5.8},
    'belgaum':      {'N': 62, 'P': 44, 'K': 42, 'ph': 6.6},
    'gulbarga':     {'N': 50, 'P': 32, 'K': 35, 'ph': 7.8},
    'davangere':    {'N': 60, 'P': 42, 'K': 40, 'ph': 6.7},
    'shimoga':      {'N': 65, 'P': 46, 'K': 48, 'ph': 6.2},
    'tumkur':       {'N': 55, 'P': 38, 'K': 36, 'ph': 6.8},
    'bidar':        {'N': 52, 'P': 35, 'K': 38, 'ph': 7.5},
    'bijapur':      {'N': 50, 'P': 33, 'K': 36, 'ph': 7.9},
    'mumbai':       {'N': 65, 'P': 55, 'K': 50, 'ph': 6.3},
    'pune':         {'N': 58, 'P': 45, 'K': 42, 'ph': 6.8},
    'nagpur':       {'N': 55, 'P': 38, 'K': 40, 'ph': 7.2},
    'nashik':       {'N': 60, 'P': 42, 'K': 45, 'ph': 6.9},
    'aurangabad':   {'N': 52, 'P': 35, 'K': 38, 'ph': 7.4},
    'solapur':      {'N': 48, 'P': 30, 'K': 35, 'ph': 7.8},
    'kolhapur':     {'N': 68, 'P': 50, 'K': 52, 'ph': 6.2},
    'amravati':     {'N': 55, 'P': 40, 'K': 42, 'ph': 7.1},
    'akola':        {'N': 52, 'P': 38, 'K': 40, 'ph': 7.3},
    'latur':        {'N': 50, 'P': 33, 'K': 36, 'ph': 7.6},
    'nanded':       {'N': 53, 'P': 35, 'K': 38, 'ph': 7.4},
    'satara':       {'N': 60, 'P': 44, 'K': 46, 'ph': 6.7},
    'delhi':        {'N': 50, 'P': 50, 'K': 45, 'ph': 7.0},
    'new delhi':    {'N': 50, 'P': 50, 'K': 45, 'ph': 7.0},
    'gurgaon':      {'N': 52, 'P': 48, 'K': 44, 'ph': 7.2},
    'gurugram':     {'N': 52, 'P': 48, 'K': 44, 'ph': 7.2},
    'faridabad':    {'N': 55, 'P': 50, 'K': 46, 'ph': 7.1},
    'chandigarh':   {'N': 60, 'P': 55, 'K': 50, 'ph': 6.9},
    'ludhiana':     {'N': 70, 'P': 60, 'K': 55, 'ph': 7.0},
    'amritsar':     {'N': 68, 'P': 58, 'K': 52, 'ph': 7.2},
    'ambala':       {'N': 62, 'P': 52, 'K': 48, 'ph': 7.0},
    'jalandhar':    {'N': 68, 'P': 56, 'K': 52, 'ph': 7.1},
    'patiala':      {'N': 65, 'P': 55, 'K': 50, 'ph': 7.2},
    'rohtak':       {'N': 55, 'P': 48, 'K': 44, 'ph': 7.5},
    'hisar':        {'N': 50, 'P': 42, 'K': 40, 'ph': 7.8},
    'panipat':      {'N': 58, 'P': 50, 'K': 46, 'ph': 7.3},
    'karnal':       {'N': 62, 'P': 52, 'K': 48, 'ph': 7.1},
    'lucknow':      {'N': 58, 'P': 48, 'K': 45, 'ph': 7.3},
    'kanpur':       {'N': 55, 'P': 45, 'K': 42, 'ph': 7.5},
    'agra':         {'N': 52, 'P': 42, 'K': 40, 'ph': 7.8},
    'varanasi':     {'N': 60, 'P': 50, 'K': 48, 'ph': 7.0},
    'allahabad':    {'N': 58, 'P': 48, 'K': 45, 'ph': 7.2},
    'prayagraj':    {'N': 58, 'P': 48, 'K': 45, 'ph': 7.2},
    'meerut':       {'N': 65, 'P': 55, 'K': 50, 'ph': 7.0},
    'mathura':      {'N': 50, 'P': 40, 'K': 38, 'ph': 8.0},
    'bareilly':     {'N': 62, 'P': 50, 'K': 48, 'ph': 7.1},
    'aligarh':      {'N': 55, 'P': 46, 'K': 44, 'ph': 7.4},
    'moradabad':    {'N': 60, 'P': 50, 'K': 46, 'ph': 7.2},
    'gorakhpur':    {'N': 62, 'P': 50, 'K': 48, 'ph': 7.0},
    'jaipur':       {'N': 45, 'P': 35, 'K': 30, 'ph': 7.5},
    'jodhpur':      {'N': 40, 'P': 30, 'K': 28, 'ph': 8.0},
    'udaipur':      {'N': 50, 'P': 38, 'K': 35, 'ph': 7.0},
    'kota':         {'N': 52, 'P': 40, 'K': 38, 'ph': 7.4},
    'bikaner':      {'N': 38, 'P': 28, 'K': 25, 'ph': 8.2},
    'ajmer':        {'N': 48, 'P': 36, 'K': 33, 'ph': 7.6},
    'alwar':        {'N': 50, 'P': 40, 'K': 36, 'ph': 7.4},
    'bharatpur':    {'N': 55, 'P': 44, 'K': 40, 'ph': 7.2},
    'ahmedabad':    {'N': 55, 'P': 42, 'K': 40, 'ph': 7.5},
    'surat':        {'N': 65, 'P': 50, 'K': 48, 'ph': 6.8},
    'vadodara':     {'N': 60, 'P': 46, 'K': 44, 'ph': 7.0},
    'rajkot':       {'N': 50, 'P': 38, 'K': 36, 'ph': 7.8},
    'gandhinagar':  {'N': 55, 'P': 42, 'K': 40, 'ph': 7.3},
    'bhavnagar':    {'N': 52, 'P': 40, 'K': 38, 'ph': 7.6},
    'jamnagar':     {'N': 50, 'P': 38, 'K': 36, 'ph': 7.8},
    'anand':        {'N': 60, 'P': 46, 'K': 44, 'ph': 7.1},
    'chennai':      {'N': 60, 'P': 40, 'K': 40, 'ph': 6.6},
    'coimbatore':   {'N': 65, 'P': 45, 'K': 48, 'ph': 6.4},
    'madurai':      {'N': 58, 'P': 38, 'K': 42, 'ph': 6.8},
    'salem':        {'N': 62, 'P': 42, 'K': 44, 'ph': 6.5},
    'tiruchirappalli': {'N': 60, 'P': 40, 'K': 42, 'ph': 6.7},
    'trichy':       {'N': 60, 'P': 40, 'K': 42, 'ph': 6.7},
    'tirunelveli':  {'N': 58, 'P': 38, 'K': 40, 'ph': 6.9},
    'vellore':      {'N': 55, 'P': 36, 'K': 38, 'ph': 7.0},
    'erode':        {'N': 62, 'P': 42, 'K': 44, 'ph': 6.5},
    'tiruppur':     {'N': 60, 'P': 40, 'K': 42, 'ph': 6.6},
    'thanjavur':    {'N': 68, 'P': 46, 'K': 50, 'ph': 6.5},
    'kolkata':      {'N': 70, 'P': 50, 'K': 45, 'ph': 6.4},
    'howrah':       {'N': 68, 'P': 48, 'K': 44, 'ph': 6.5},
    'durgapur':     {'N': 65, 'P': 46, 'K': 42, 'ph': 6.6},
    'siliguri':     {'N': 72, 'P': 52, 'K': 48, 'ph': 6.0},
    'asansol':      {'N': 65, 'P': 46, 'K': 43, 'ph': 6.5},
    'bhopal':       {'N': 55, 'P': 42, 'K': 40, 'ph': 7.2},
    'indore':       {'N': 58, 'P': 44, 'K': 42, 'ph': 7.0},
    'gwalior':      {'N': 52, 'P': 40, 'K': 38, 'ph': 7.5},
    'jabalpur':     {'N': 60, 'P': 46, 'K': 44, 'ph': 6.8},
    'ujjain':       {'N': 55, 'P': 42, 'K': 40, 'ph': 7.3},
    'ratlam':       {'N': 52, 'P': 38, 'K': 38, 'ph': 7.4},
    'thiruvananthapuram': {'N': 72, 'P': 52, 'K': 55, 'ph': 5.8},
    'trivandrum':   {'N': 72, 'P': 52, 'K': 55, 'ph': 5.8},
    'kochi':        {'N': 75, 'P': 55, 'K': 58, 'ph': 5.7},
    'cochin':       {'N': 75, 'P': 55, 'K': 58, 'ph': 5.7},
    'kozhikode':    {'N': 70, 'P': 50, 'K': 55, 'ph': 5.9},
    'calicut':      {'N': 70, 'P': 50, 'K': 55, 'ph': 5.9},
    'thrissur':     {'N': 72, 'P': 52, 'K': 56, 'ph': 5.8},
    'kollam':       {'N': 70, 'P': 50, 'K': 54, 'ph': 5.9},
    'palakkad':     {'N': 65, 'P': 46, 'K': 50, 'ph': 6.2},
    'bhubaneswar':  {'N': 65, 'P': 45, 'K': 48, 'ph': 6.5},
    'cuttack':      {'N': 68, 'P': 48, 'K': 50, 'ph': 6.3},
    'rourkela':     {'N': 60, 'P': 42, 'K': 45, 'ph': 6.8},
    'patna':        {'N': 62, 'P': 50, 'K': 48, 'ph': 7.0},
    'gaya':         {'N': 58, 'P': 46, 'K': 44, 'ph': 7.2},
    'muzaffarpur':  {'N': 65, 'P': 52, 'K': 50, 'ph': 6.8},
    'bhagalpur':    {'N': 62, 'P': 50, 'K': 46, 'ph': 6.9},
    'ranchi':       {'N': 58, 'P': 40, 'K': 42, 'ph': 6.5},
    'dhanbad':      {'N': 55, 'P': 38, 'K': 40, 'ph': 6.8},
    'jamshedpur':   {'N': 60, 'P': 42, 'K': 44, 'ph': 6.6},
    'raipur':       {'N': 62, 'P': 44, 'K': 46, 'ph': 6.7},
    'bhilai':       {'N': 60, 'P': 42, 'K': 44, 'ph': 6.8},
    'bilaspur':     {'N': 58, 'P': 40, 'K': 42, 'ph': 6.9},
    'shimla':       {'N': 48, 'P': 45, 'K': 50, 'ph': 6.0},
    'dharamshala':  {'N': 50, 'P': 46, 'K': 52, 'ph': 5.8},
    'manali':       {'N': 45, 'P': 42, 'K': 48, 'ph': 6.2},
    'srinagar':     {'N': 45, 'P': 42, 'K': 48, 'ph': 6.5},
    'jammu':        {'N': 52, 'P': 46, 'K': 50, 'ph': 7.0},
    'guwahati':     {'N': 68, 'P': 48, 'K': 52, 'ph': 5.8},
    'dibrugarh':    {'N': 70, 'P': 50, 'K': 55, 'ph': 5.6},
    'silchar':      {'N': 65, 'P': 46, 'K': 50, 'ph': 5.9},
    'dehradun':     {'N': 55, 'P': 48, 'K': 50, 'ph': 6.5},
    'haridwar':     {'N': 58, 'P': 50, 'K': 52, 'ph': 7.0},
    'rishikesh':    {'N': 52, 'P': 46, 'K': 48, 'ph': 6.7},
    'port blair':   {'N': 72, 'P': 55, 'K': 60, 'ph': 5.5},
    'default':      {'N': 50, 'P': 50, 'K': 50, 'ph': 6.5},
}

# ─────────────────────────────────────────────
#  ANNUAL RAINFALL LOOKUP TABLE (mm/year)
#  Used to supplement/correct NASA daily data
# ─────────────────────────────────────────────
ANNUAL_RAINFALL_DB = {
    # Telangana & AP
    'hyderabad': 812,  'warangal': 1100, 'nizamabad': 960,
    'karimnagar': 1020, 'khammam': 1080, 'nalgonda': 780,
    'mahbubnagar': 720, 'adilabad': 1100, 'medak': 870,
    'vijayawada': 1000, 'visakhapatnam': 1100, 'guntur': 950,
    'kurnool': 680,    'tirupati': 920,  'rajahmundry': 1100,
    'nellore': 950,    'kadapa': 780,    'anantapur': 550,
    'chittoor': 850,
    # Karnataka
    'bangalore': 970,  'bengaluru': 970, 'mysore': 780,
    'hubli': 820,      'mangalore': 3200,'belgaum': 980,
    'gulbarga': 720,   'davangere': 650, 'shimoga': 1800,
    'tumkur': 680,     'bidar': 820,     'bijapur': 580,
    # Maharashtra
    'mumbai': 2200,    'pune': 720,      'nagpur': 1200,
    'nashik': 750,     'aurangabad': 720,'solapur': 550,
    'kolhapur': 1050,  'amravati': 1050, 'akola': 860,
    'latur': 650,      'nanded': 850,    'satara': 850,
    # North India
    'delhi': 780,      'new delhi': 780, 'gurgaon': 750,
    'gurugram': 750,   'faridabad': 760, 'chandigarh': 1110,
    'ludhiana': 720,   'amritsar': 680,  'ambala': 830,
    'jalandhar': 690,  'patiala': 720,   'rohtak': 630,
    'hisar': 430,      'panipat': 620,   'karnal': 740,
    # UP
    'lucknow': 900,    'kanpur': 830,    'agra': 680,
    'varanasi': 1100,  'allahabad': 1020,'prayagraj': 1020,
    'meerut': 850,     'mathura': 640,   'bareilly': 980,
    'aligarh': 780,    'moradabad': 960, 'gorakhpur': 1100,
    # Rajasthan
    'jaipur': 590,     'jodhpur': 370,   'udaipur': 640,
    'kota': 720,       'bikaner': 280,   'ajmer': 490,
    'alwar': 680,      'bharatpur': 680,
    # Gujarat
    'ahmedabad': 780,  'surat': 1400,    'vadodara': 920,
    'rajkot': 600,     'gandhinagar': 820,'bhavnagar': 580,
    'jamnagar': 480,   'anand': 870,
    # Tamil Nadu
    'chennai': 1400,   'coimbatore': 680,'madurai': 850,
    'salem': 920,      'tiruchirappalli': 840,'trichy': 840,
    'tirunelveli': 730,'vellore': 1020,  'erode': 780,
    'tiruppur': 720,   'thanjavur': 1100,
    # West Bengal
    'kolkata': 1600,   'howrah': 1600,   'durgapur': 1400,
    'siliguri': 2700,  'asansol': 1300,
    # MP
    'bhopal': 1180,    'indore': 1020,   'gwalior': 820,
    'jabalpur': 1400,  'ujjain': 820,    'ratlam': 820,
    # Kerala
    'thiruvananthapuram': 1600,'trivandrum': 1600,'kochi': 3100,
    'cochin': 3100,    'kozhikode': 2900,'calicut': 2900,
    'thrissur': 2900,  'kollam': 1700,   'palakkad': 2000,
    # Odisha
    'bhubaneswar': 1500,'cuttack': 1500, 'rourkela': 1400,
    # Bihar
    'patna': 1100,     'gaya': 1050,     'muzaffarpur': 1200,
    'bhagalpur': 1100,
    # Jharkhand
    'ranchi': 1400,    'dhanbad': 1300,  'jamshedpur': 1400,
    # Chhattisgarh
    'raipur': 1300,    'bhilai': 1300,   'bilaspur': 1200,
    # HP & J&K
    'shimla': 1600,    'dharamshala': 2900,'manali': 1100,
    'srinagar': 650,   'jammu': 1050,
    # Northeast
    'guwahati': 1600,  'dibrugarh': 2900,'silchar': 3200,
    # Uttarakhand
    'dehradun': 2100,  'haridwar': 1800, 'rishikesh': 1900,
    # Islands
    'port blair': 3200,
    # Default fallback
    'default': 900,
}


# ─────────────────────────────────────────────
#  EXTENDED CROP DATABASE — 50+ crops
# ─────────────────────────────────────────────
EXTENDED_CROPS = {
    # Cereals & Millets
    "wheat":         {"temp": (5, 25),  "rain": (30, 100),  "moist": (25, 55), "ph": (6.0, 7.5), "regions": ["north", "central", "plains"]},
    "barley":        {"temp": (5, 22),  "rain": (25, 80),   "moist": (20, 50), "ph": (6.0, 7.5), "regions": ["north", "northwest"]},
    "sorghum":       {"temp": (25, 38), "rain": (40, 100),  "moist": (20, 50), "ph": (5.5, 8.0), "regions": ["deccan", "central", "south"]},
    "bajra":         {"temp": (25, 40), "rain": (25, 80),   "moist": (15, 45), "ph": (6.0, 8.0), "regions": ["northwest", "west", "deccan"]},
    "ragi":          {"temp": (20, 35), "rain": (60, 150),  "moist": (30, 55), "ph": (5.5, 7.5), "regions": ["south", "deccan", "eastern"]},
    "oats":          {"temp": (5, 20),  "rain": (50, 120),  "moist": (35, 60), "ph": (5.5, 7.0), "regions": ["north", "hills"]},
    # Oilseeds
    "mustard":       {"temp": (10, 25), "rain": (20, 70),   "moist": (20, 45), "ph": (6.0, 7.5), "regions": ["north", "central", "plains"]},
    "groundnut":     {"temp": (22, 35), "rain": (50, 120),  "moist": (25, 55), "ph": (6.0, 7.5), "regions": ["south", "deccan", "west"]},
    "sunflower":     {"temp": (20, 35), "rain": (40, 100),  "moist": (25, 50), "ph": (6.0, 7.5), "regions": ["deccan", "central", "south"]},
    "sesame":        {"temp": (25, 38), "rain": (40, 80),   "moist": (20, 45), "ph": (5.5, 7.5), "regions": ["south", "deccan", "east"]},
    "soybean":       {"temp": (20, 35), "rain": (60, 150),  "moist": (35, 60), "ph": (6.0, 7.0), "regions": ["central", "deccan", "north"]},
    "castor":        {"temp": (25, 40), "rain": (40, 80),   "moist": (20, 45), "ph": (6.0, 8.0), "regions": ["deccan", "south", "west"]},
    # Cash Crops
    "sugarcane":     {"temp": (22, 38), "rain": (100, 250), "moist": (45, 75), "ph": (6.0, 7.5), "regions": ["south", "deccan", "plains", "west"]},
    "tobacco":       {"temp": (18, 35), "rain": (50, 120),  "moist": (30, 55), "ph": (5.5, 7.0), "regions": ["south", "deccan", "east"]},
    # Spices
    "turmeric":      {"temp": (20, 35), "rain": (150, 300), "moist": (55, 80), "ph": (5.0, 7.0), "regions": ["south", "east", "deccan"]},
    "ginger":        {"temp": (18, 30), "rain": (150, 300), "moist": (55, 80), "ph": (5.5, 6.5), "regions": ["south", "east", "hills"]},
    "chilli":        {"temp": (20, 35), "rain": (60, 150),  "moist": (35, 60), "ph": (6.0, 7.0), "regions": ["south", "deccan", "east"]},
    "coriander":     {"temp": (15, 30), "rain": (30, 80),   "moist": (25, 50), "ph": (6.0, 7.5), "regions": ["central", "north", "deccan"]},
    "cumin":         {"temp": (15, 30), "rain": (20, 60),   "moist": (15, 40), "ph": (6.5, 8.0), "regions": ["northwest", "west"]},
    "cardamom":      {"temp": (15, 25), "rain": (250, 400), "moist": (70, 90), "ph": (5.0, 6.5), "regions": ["south_hills"]},
    "pepper":        {"temp": (20, 35), "rain": (200, 400), "moist": (65, 85), "ph": (5.0, 6.5), "regions": ["south", "south_hills"]},
    # Vegetables
    "tomato":        {"temp": (18, 30), "rain": (60, 120),  "moist": (35, 60), "ph": (5.5, 7.0), "regions": ["all"]},
    "potato":        {"temp": (10, 20), "rain": (50, 100),  "moist": (40, 60), "ph": (5.0, 6.5), "regions": ["north", "hills", "central"]},
    "onion":         {"temp": (13, 28), "rain": (40, 90),   "moist": (25, 50), "ph": (6.0, 7.5), "regions": ["all"]},
    "cabbage":       {"temp": (10, 22), "rain": (50, 100),  "moist": (35, 60), "ph": (6.0, 7.0), "regions": ["north", "hills", "south"]},
    "cauliflower":   {"temp": (10, 20), "rain": (50, 100),  "moist": (35, 60), "ph": (6.0, 7.0), "regions": ["north", "hills", "south"]},
    "brinjal":       {"temp": (18, 35), "rain": (50, 120),  "moist": (30, 55), "ph": (5.5, 7.0), "regions": ["all"]},
    "okra":          {"temp": (20, 38), "rain": (50, 120),  "moist": (30, 55), "ph": (6.0, 7.5), "regions": ["all"]},
    # Fruits
    "guava":         {"temp": (20, 38), "rain": (50, 120),  "moist": (30, 55), "ph": (5.0, 7.0), "regions": ["all"]},
    "litchi":        {"temp": (18, 28), "rain": (100, 200), "moist": (50, 70), "ph": (5.5, 7.0), "regions": ["east", "north"]},
    "pineapple":     {"temp": (20, 35), "rain": (150, 300), "moist": (60, 80), "ph": (4.5, 6.5), "regions": ["south", "east", "northeast"]},
    "jackfruit":     {"temp": (22, 35), "rain": (150, 300), "moist": (55, 80), "ph": (5.0, 7.0), "regions": ["south", "east"]},
    "tamarind":      {"temp": (25, 40), "rain": (40, 100),  "moist": (20, 50), "ph": (6.0, 7.5), "regions": ["south", "deccan"]},
    "jamun":         {"temp": (22, 38), "rain": (60, 150),  "moist": (35, 60), "ph": (5.5, 7.0), "regions": ["all"]},
    # Plantation
    "tea":           {"temp": (15, 28), "rain": (200, 400), "moist": (65, 85), "ph": (4.5, 6.0), "regions": ["northeast", "hills", "south_hills"]},
    "rubber":        {"temp": (22, 35), "rain": (250, 400), "moist": (70, 90), "ph": (4.5, 6.0), "regions": ["south", "south_hills", "northeast"]},
    "arecanut":      {"temp": (22, 35), "rain": (150, 300), "moist": (60, 80), "ph": (5.0, 7.0), "regions": ["south", "south_hills", "east"]},
    # Other high-value
    "garlic":        {"temp": (12, 25), "rain": (30, 80),   "moist": (25, 50), "ph": (6.0, 7.5), "regions": ["north", "central", "all"]},
    "peas":          {"temp": (5, 22),  "rain": (40, 90),   "moist": (35, 60), "ph": (6.0, 7.5), "regions": ["north", "hills", "central"]},
    "cucumber":      {"temp": (18, 35), "rain": (50, 120),  "moist": (35, 60), "ph": (5.5, 7.0), "regions": ["all"]},
    "bitter_gourd":  {"temp": (20, 38), "rain": (50, 120),  "moist": (30, 55), "ph": (6.0, 7.0), "regions": ["all"]},
    "drumstick":     {"temp": (25, 40), "rain": (30, 100),  "moist": (20, 50), "ph": (6.0, 8.0), "regions": ["south", "deccan"]},
    "aloe_vera":     {"temp": (20, 40), "rain": (20, 80),   "moist": (15, 45), "ph": (7.0, 8.5), "regions": ["northwest", "west", "deccan"]},
    "bamboo":        {"temp": (18, 38), "rain": (100, 300), "moist": (50, 75), "ph": (5.0, 6.5), "regions": ["east", "northeast", "south"]},
    "hemp":          {"temp": (12, 28), "rain": (50, 100),  "moist": (30, 55), "ph": (6.0, 7.5), "regions": ["north", "hills"]},
    "flax":          {"temp": (10, 22), "rain": (40, 90),   "moist": (30, 55), "ph": (5.5, 7.0), "regions": ["north", "central"]},
    "saffron":       {"temp": (5, 20),  "rain": (30, 80),   "moist": (20, 50), "ph": (6.0, 8.0), "regions": ["hills", "northwest"]},
    "chrysanthemum": {"temp": (10, 25), "rain": (60, 120),  "moist": (35, 55), "ph": (5.5, 7.0), "regions": ["south", "hills", "all"]},
    "rose":          {"temp": (15, 28), "rain": (50, 100),  "moist": (35, 60), "ph": (6.0, 7.0), "regions": ["all"]},
    "marigold":      {"temp": (18, 35), "rain": (40, 100),  "moist": (25, 55), "ph": (5.5, 7.5), "regions": ["all"]},
}

EXTENDED_CROP_EMOJI = {
    "wheat": "🌾", "barley": "🌾", "sorghum": "🌾", "bajra": "🌾", "ragi": "🌾",
    "oats": "🌾", "mustard": "🌻", "groundnut": "🥜", "sunflower": "🌻",
    "sesame": "🌿", "soybean": "🫘", "castor": "🌿", "sugarcane": "🎋",
    "tobacco": "🍃", "turmeric": "🟡", "ginger": "🫚",
    "chilli": "🌶️", "coriander": "🌿", "cumin": "🌿", "cardamom": "🫚",
    "pepper": "🫙", "tomato": "🍅", "potato": "🥔", "onion": "🧅",
    "cabbage": "🥬", "cauliflower": "🥦", "brinjal": "🍆", "okra": "🌿",
    "guava": "🍈", "litchi": "🍒", "pineapple": "🍍", "jackfruit": "🍈",
    "tamarind": "🫒", "jamun": "🍇", "tea": "🍵", "rubber": "🌿",
    "arecanut": "🥥", "garlic": "🧄", "peas": "🫛", "cucumber": "🥒",
    "bitter_gourd": "🥒", "drumstick": "🌿", "aloe_vera": "🌵", "bamboo": "🎋",
    "hemp": "🌿", "flax": "🌿", "saffron": "🌸", "chrysanthemum": "🌼",
    "rose": "🌹", "marigold": "🌻",
}


def get_climate_zone(lat, lon, temp, rain):
    """Return list of climate zone labels based on coordinates + weather."""
    zones = []
    if lat > 28 and (lon < 78 or lon > 88):
        zones.append("hills")
    if lat > 27:
        zones.append("north")
    if lat < 15:
        zones.append("south")
    if 10 <= lat <= 18 and 74 <= lon <= 77:
        zones.append("south_hills")
    if lat > 24 and lon > 88:
        zones.append("northeast")
    if lon > 85:
        zones.append("east")
    if lon < 75 and lat > 22:
        zones.append("northwest")
    if lon < 76 and lat < 24:
        zones.append("west")
    if 15 < lat < 22 and 74 < lon < 84:
        zones.append("deccan")
    if 22 < lat < 28 and 74 < lon < 82:
        zones.append("central")
    if 25 < lat < 30 and 75 < lon < 85:
        zones.append("plains")
    zones.append("all")
    return list(set(zones))


def get_extended_recommendations(lat, lon, temp, rain, moist, ph, top_ml_crops):
    """
    Score additional crops from EXTENDED_CROPS based on location + weather + soil.
    Returns up to 8 additional recommendations not already in ML top-5.
    """
    zones = get_climate_zone(lat, lon, temp, rain)
    ml_names = {c["name"].lower() for c in top_ml_crops}
    scored = []

    for crop, params in EXTENDED_CROPS.items():
        if crop in ml_names:
            continue
        if not any(r in zones for r in params["regions"]):
            continue

        score = 0.0

        t_min, t_max = params["temp"]
        if t_min <= temp <= t_max:
            score += 30
        elif abs(temp - (t_min + t_max) / 2) < 8:
            score += 15

        r_min, r_max = params["rain"]
        if r_min <= rain <= r_max:
            score += 25
        elif abs(rain - r_min) < 20 or abs(rain - r_max) < 20:
            score += 10

        m_min, m_max = params["moist"]
        if m_min <= moist <= m_max:
            score += 25
        elif abs(moist - (m_min + m_max) / 2) < 12:
            score += 10

        ph_min, ph_max = params["ph"]
        if ph_min <= ph <= ph_max:
            score += 20
        elif abs(ph - (ph_min + ph_max) / 2) < 0.8:
            score += 8

        if score >= 45:
            scored.append({
                "name": crop,
                "raw_score": score,
                "emoji": EXTENDED_CROP_EMOJI.get(crop, "🌱"),
                "source": "region+climate"
            })

    scored.sort(key=lambda x: x["raw_score"], reverse=True)

    result = []
    for i, c in enumerate(scored[:8]):
        norm_conf = round(44 - (i * 3), 1)
        result.append({
            "rank":       i + 6,
            "name":       c["name"],
            "confidence": norm_conf,
            "emoji":      c["emoji"],
            "source":     "region+climate"
        })
    return result


def get_soil(city_name: str) -> dict:
    """Return soil values for a city (case-insensitive). Falls back to default."""
    key = city_name.strip().lower()
    if key in SOIL_DB:
        return SOIL_DB[key]
    for db_key in SOIL_DB:
        if db_key in key or key in db_key:
            return SOIL_DB[db_key]
    return SOIL_DB['default']


def get_annual_rainfall(city_name: str) -> float:
    """Return known annual rainfall (mm/year) for the city."""
    key = city_name.strip().lower()
    if key in ANNUAL_RAINFALL_DB:
        return float(ANNUAL_RAINFALL_DB[key])
    for db_key in ANNUAL_RAINFALL_DB:
        if db_key in key or key in db_key:
            return float(ANNUAL_RAINFALL_DB[db_key])
    return float(ANNUAL_RAINFALL_DB['default'])


TEXT_MAP = {
    "English": {
        "title": "Smart Farmer AI", "village_lbl": "Village Name", "btn": "Predict Crop",
        "opt1": "Crop Prediction", "opt2": "Market Prices", "opt3": "Leaf Analysis",
        "mandi_search_title": "Mandi Market Prices", "audio_btn": "Listen"
    },
    "Telugu": {
        "title": "స్మార్ట్ రైతు AI", "village_lbl": "గ్రామం పేరు", "btn": "పంటను అంచనా వేయండి",
        "opt1": "పంట అంచనా", "opt2": "మార్కెట్ ధరలు", "opt3": "ఆకు విశ్లేషణ",
        "mandi_search_title": "మండి మార్కెట్ ధరలు", "audio_btn": "వినండి"
    },
    "Hindi": {
        "title": "स्मार्ट किसान AI", "village_lbl": "गाँव का नाम", "btn": "फसल की भविष्यवाणी",
        "opt1": "फसल भविष्यवाणी", "opt2": "मंडी भाव", "opt3": "पत्ती विश्लेषण",
        "mandi_search_title": "मंडी बाजार भाव", "audio_btn": "सुनें"
    }
}

CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🌿", "mothbeans": "🌱", "mungbean": "🌱", "blackgram": "🌱",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "cotton": "🌿",
    "jute": "🌿", "coffee": "☕"
}


def call_groq(prompt, model_override=None):
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body    = {
        "model":       model_override or GROQ_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  2048,
        "temperature": 0.7
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_groq_vision(image_bytes, mime_type, prompt):
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    b64     = base64.b64encode(image_bytes).decode("utf-8")
    body    = {
        "model": GROQ_VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            {"type": "text",      "text": prompt}
        ]}],
        "max_tokens": 1024, "temperature": 0.5
    }
    resp = requests.post(url, headers=headers, json=body, timeout=40)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def get_nasa_data(lat, lon):
    """Fetch real rainfall and soil moisture from NASA POWER API (daily values)."""
    date = (datetime.now() - timedelta(days=4)).strftime('%Y%m%d')
    url  = (f"https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=PRECTOTCORR,GWETTOP&community=AG"
            f"&longitude={lon}&latitude={lat}&start={date}&end={date}&format=JSON")
    try:
        res       = requests.get(url, timeout=12).json()
        rain_raw  = list(res['properties']['parameter']['PRECTOTCORR'].values())[0]
        moist_raw = list(res['properties']['parameter']['GWETTOP'].values())[0]
        # GWETTOP (0–1 fraction) → percent
        moist = round(float(moist_raw) * 100, 2)
        # ── CRITICAL FIX ──
        # PRECTOTCORR is mm/day. The ML model was trained on annual rainfall (20–300 mm).
        # We return the raw daily value here; the caller will use the city DB for annual rain.
        daily_rain = round(float(rain_raw), 2)
        return daily_rain, moist
    except Exception:
        return None, 45.0          # None signals: use DB fallback for rain


# ───────────── ROUTES ─────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/input", methods=["POST"])
def input_page():
    lang = request.form.get("language", "English")
    return render_template("input.html", lang=lang, t=TEXT_MAP[lang])


@app.route("/predict", methods=["POST"])
def predict():
    city   = request.form.get("city", "").strip()
    lang   = request.form.get("lang", "English")
    labels = TEXT_MAP.get(lang, TEXT_MAP["English"])

    if not city:
        return "<h2 style='color:red;text-align:center;margin-top:50px'>⚠️ Please enter a village/city name.</h2>"

    geolocator = Nominatim(user_agent="agri_pro_ambassador")
    location   = geolocator.geocode(f"{city}, India", addressdetails=True)
    if not location:
        return "<h2 style='color:red;text-align:center;margin-top:50px'>⚠️ Location Not Found.</h2>"

    lat, lon = location.latitude, location.longitude

    # ── [1] NASA POWER API: daily rainfall & soil moisture ──
    daily_rain, moist = get_nasa_data(lat, lon)

    # ── [2] OpenWeatherMap API: real temperature, humidity, wind ──
    weather_url = (f"http://api.openweathermap.org/data/2.5/weather"
                   f"?lat={lat}&lon={lon}"
                   f"&appid={WEATHER_API_KEY}&units=metric")
    try:
        w_res        = requests.get(weather_url, timeout=10).json()
        temp         = round(w_res['main']['temp'], 1)
        hum          = w_res['main']['humidity']
        weather_desc = w_res.get('weather', [{}])[0].get('description', '').title()
        wind_speed   = round(w_res.get('wind', {}).get('speed', 0), 1)
    except Exception as e:
        return f"<h2 style='color:red;text-align:center'>⚠️ Weather API Error: {e}</h2>"

    # ── [3] ICAR Soil DB: city-specific N, P, K, pH ──
    soil = get_soil(city)
    N, P, K, ph = soil['N'], soil['P'], soil['K'], soil['ph']

    # ── [4] RAINFALL: Use city annual DB as primary source ──
    #
    # WHY: The ML model was trained on annual rainfall values (20–300 mm).
    # NASA daily rainfall (0–15 mm/day) is a completely different scale and
    # causes the model to always predict muskmelon (lowest rain crop: 24 mm).
    # We use our curated annual rainfall DB as the primary source for the ML
    # model. NASA daily rain is kept separately just for display purposes.
    #
    rain = get_annual_rainfall(city)   # ← annual mm, matches training data range

    # ── [5] XGBoost ML Model prediction ──
    # Feature order MUST match train_model.py:
    # ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    features   = np.array([[N, P, K, temp, hum, ph, rain]])
    probs      = model.predict_proba(features)[0]
    all_ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

    print(f"\n📍 {city} | N={N} P={P} K={K} temp={temp} hum={hum} ph={ph} rain={rain}")
    print(f"   Top-3 predictions:")
    for idx, prob in all_ranked[:3]:
        print(f"     {le.classes_[idx]:15s}: {prob*100:.1f}%")

    top5_crops = []
    for rank, (idx, prob) in enumerate(all_ranked[:5]):
        crop_name  = le.classes_[idx]
        confidence = round(float(prob) * 100, 1)
        top5_crops.append({
            "rank":       rank + 1,
            "name":       crop_name,
            "confidence": confidence,
            "emoji":      CROP_EMOJI.get(crop_name, "🌱"),
            "source":     "ml_model"
        })

    # ── [6] Climate + Region Extended Recommendations ──
    extended = get_extended_recommendations(lat, lon, temp, rain, moist, ph, top5_crops)

    # Climate zone for display
    zones        = get_climate_zone(lat, lon, temp, rain)
    zone_display = ", ".join([z.replace("_", " ").title() for z in zones if z != "all"][:3])

    best_crop = top5_crops[0]["name"]

    # Use daily rain for display (more informative), annual rain for ML
    display_rain = round(daily_rain, 1) if daily_rain is not None else round(rain / 120, 1)

    return render_template("result.html",
                           crop=best_crop,
                           top5=top5_crops,
                           extended=extended,
                           city=city.title(),
                           temp=temp, hum=hum,
                           rain=display_rain,        # daily mm shown on card
                           annual_rain=round(rain),  # annual mm used by model
                           moist=moist,
                           weather_desc=weather_desc, wind_speed=wind_speed,
                           N=N, P=P, K=K, ph=ph,
                           lang=lang, t=labels,
                           lat=lat, lon=lon,
                           climate_zone=zone_display)


@app.route("/get_advice", methods=["POST"])
def get_advice():
    d       = request.get_json()
    section = d.get("section")
    crop    = d.get("crop")
    city    = d.get("city")
    lang    = d.get("lang", "English")
    temp    = d.get("temp")
    hum     = d.get("hum")
    rain    = d.get("rain")
    moist   = d.get("moist")

    prompts = {
        "FERT": f"""You are an expert agricultural officer. Crop: {crop} in {city}, India.
Conditions — Temp: {temp}°C, Humidity: {hum}%, Rainfall: {rain}mm, Soil Moisture: {moist}%.
Reply ONLY in {lang} using ONLY valid HTML tags. No markdown, no asterisks, no backticks.

<h4>NPK Fertilizer Schedule</h4>
<table><tr><th>Growth Stage</th><th>Fertilizer Name</th><th>Quantity per Acre</th><th>Timing</th></tr>
...at least 4 rows...</table>
<h4>Organic Amendments</h4>
<ul><li>...</li><li>...</li><li>...</li></ul>
<h4>Application Tips for {city}</h4>
<ul><li>...</li><li>...</li><li>...</li></ul>""",

        "WATER": f"""You are an expert agronomist. Crop: {crop} in {city}, India.
Conditions — Temp: {temp}°C, Humidity: {hum}%, Rainfall: {rain}mm, Soil Moisture: {moist}%.
Reply ONLY in {lang} using ONLY valid HTML tags. No markdown, no asterisks, no backticks.

<h4>Water Requirement by Growth Stage</h4>
<table><tr><th>Growth Stage</th><th>Water Need (mm)</th><th>Irrigation Frequency</th></tr>
...at least 4 rows...</table>
<h4>Recommended Irrigation Methods for {city}</h4>
<table><tr><th>Method</th><th>Pros</th><th>Best For</th></tr>
...at least 3 rows...</table>
<h4>Water Saving Tips</h4>
<ul><li>...</li><li>...</li><li>...</li></ul>""",

        "CALENDAR": f"""You are a local agri-officer for {city}, India. Crop: {crop}.
Reply ONLY in {lang} using ONLY valid HTML tags. No markdown, no asterisks, no backticks.

<h4>12-Month Crop Calendar — {crop} in {city}</h4>
<table><tr><th>Month</th><th>Activity</th><th>Key Task</th><th>Alert / Warning</th></tr>
...one row per month, ALL 12 months January to December...</table>
<h4>Critical Dates for {city}</h4>
<ul><li>Sowing window: ...</li><li>Transplanting: ...</li><li>Flowering: ...</li><li>Harvest: ...</li></ul>""",

        "MANDI": f"""You are an Indian agri commodity market analyst. Crop: {crop}, near {city}, India.
Reply ONLY in {lang} using ONLY valid HTML tags. No markdown, no asterisks, no backticks.

<h4>Nearby Mandi Prices — {crop}</h4>
<table><tr><th>Mandi Name</th><th>District</th><th>Min ₹/quintal</th><th>Max ₹/quintal</th><th>Modal Price</th><th>Trend</th></tr>
...at least 5 real mandis near {city}...</table>
<h4>Price Forecast — Next 30 Days</h4>
<p>...detailed analysis...</p>
<h4>Best Selling Strategy</h4>
<ul><li>...</li><li>...</li><li>...</li></ul>
<h4>Government MSP</h4>
<p>Current MSP for {crop}: ₹... per quintal.</p>"""
    }

    prompt = prompts.get(section)
    if not prompt:
        return jsonify({"error": "Invalid section"}), 400

    try:
        html = call_groq(prompt)
        html = html.replace("```html","").replace("```","").replace("**","").strip()
        return jsonify({"html": html})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/mandi_search", methods=["POST"])
def mandi_search():
    crop_query = request.form.get("crop_query", "").strip()
    lang       = request.form.get("lang", "English")
    t          = TEXT_MAP.get(lang, TEXT_MAP["English"])
    result     = None

    if crop_query:
        prompt = f"""You are an Indian agri commodity market analyst.
Crop: {crop_query}, India. Reply ONLY in {lang} using plain text only.
Give: 5 mandis with min/max/modal price in Rs/quintal, trend, best time to sell, MSP, 30-day forecast."""
        try:
            result = call_groq(prompt)
            result = result.replace("```","").replace("**","").strip()
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("mandi.html", lang=lang, t=t, query=crop_query, result=result)


@app.route("/analyze_disease", methods=["POST"])
def analyze_disease():
    lang      = request.form.get("lang", "English")
    t         = TEXT_MAP.get(lang, TEXT_MAP["English"])
    leaf_file = request.files.get("leaf_image")

    if not leaf_file:
        return render_template("disease.html", lang=lang, t=t,
                               result="<p style='color:red'>⚠️ No image uploaded.</p>")

    image_bytes = leaf_file.read()
    mime_type   = leaf_file.content_type or "image/jpeg"

    prompt = f"""You are an expert plant pathologist. Analyze this leaf image.
Reply ONLY in {lang} using ONLY valid HTML. No markdown or backticks.

<h4>Disease Detected</h4><p>...</p>
<h4>Symptoms Observed</h4><ul><li>...</li></ul>
<h4>Cause</h4><p>...</p>
<h4>Treatment Plan</h4>
<table><tr><th>Type</th><th>Product</th><th>Dosage</th></tr>
<tr><td>Chemical</td><td>...</td><td>...</td></tr>
<tr><td>Organic</td><td>...</td><td>...</td></tr>
<tr><td>Preventive</td><td>...</td><td>...</td></tr></table>
<h4>Severity Level</h4><p>Low/Medium/High — with explanation</p>
<h4>Farmer Action Steps</h4><ul><li>...</li><li>...</li><li>...</li></ul>"""

    try:
        html = call_groq_vision(image_bytes, mime_type, prompt)
        html = html.replace("```html","").replace("```","").replace("**","").strip()
    except Exception as e:
        html = f"<p style='color:red'>⚠️ Analysis failed: {str(e)}</p>"

    return render_template("disease.html", lang=lang, t=t, result=html)


if __name__ == "__main__":
    app.run(debug=True)