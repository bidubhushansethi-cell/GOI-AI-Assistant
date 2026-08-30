import ast
import io
import json
import operator as op
import os
import pickle
import random
import re
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import requests
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.saving import register_keras_serializable


# ============================================================
# COMPATIBILITY LAYERS (FIXED)
# ============================================================
# These classes let an old-format saved model (.h5) load cleanly
# on newer TensorFlow/Keras versions. Each one now properly
# implements get_config() / from_config() so Keras can correctly
# serialize/deserialize nested objects (like initializers)
# instead of crashing with "could not be deserialized properly".

@register_keras_serializable(package="Custom", name="GlorotUniform")
class CompatibleGlorotUniform(GlorotUniform):
    def __init__(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        super().__init__(seed=seed)

    def get_config(self):
        return {"seed": self.seed}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("input_axes", None)
        config.pop("output_axes", None)
        return cls(**config)


@register_keras_serializable(package="Custom", name="Dense")
class CompatibleDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop("quantization_config", None)
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return cls(**config)


@register_keras_serializable(package="Custom", name="InputLayer")
class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        batch_shape = kwargs.pop("batch_shape", None)
        kwargs.pop("optional", None)

        if batch_shape is not None:
            kwargs["batch_size"] = batch_shape[0]
            kwargs["shape"] = tuple(batch_shape[1:])

        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("optional", None)
        return cls(**config)


# ============================================================
# OWNER INFORMATION
# ============================================================

OWNER_NAME = "BIDUVUSHAN SETHI"
OWNER_DOB = "06 February 2006"
OWNER_BIRTH_DATE = date(2006, 2, 6)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="GO! AI - Smart Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(99,102,241,0.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 0%,
            rgba(6,182,212,0.13),
            transparent 28%
        ),
        #080c16;
    color: #e5e7eb;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.2rem;
    padding-bottom: 6rem;
}

header[data-testid="stHeader"] {
    background: rgba(8,12,22,0.8);
}

section[data-testid="stSidebar"] {
    background: #0c1220;
    border-right: 1px solid rgba(255,255,255,0.07);
}

.hero {
    border: 1px solid rgba(129,140,248,0.20);
    border-radius: 24px;
    padding: 28px 30px;
    background:
        linear-gradient(
            135deg,
            rgba(79,70,229,0.22),
            rgba(8,145,178,0.08)
        );
    box-shadow: 0 25px 70px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 7px;
    line-height: 1.6;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 15px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(34,197,94,0.10);
    border: 1px solid rgba(34,197,94,0.22);
    color: #86efac;
    font-size: 12px;
    font-weight: 600;
}

.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 12px rgba(34,197,94,0.8);
}

div[data-testid="stChatMessage"] {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(17,24,39,0.72);
    margin-bottom: 10px;
}

div[data-testid="stChatInput"] textarea {
    background: #111827 !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 17px !important;
}

.tool-card {
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    background: rgba(17,24,39,0.72);
    padding: 17px;
    height: 100%;
}

.tool-icon {
    font-size: 23px;
}

.tool-name {
    color: #f8fafc;
    font-weight: 700;
    margin-top: 7px;
}

.tool-example {
    color: #8492a8;
    font-size: 11px;
    margin-top: 5px;
}

.stat {
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    background: #111827;
    padding: 12px;
    text-align: center;
}

.stat-number {
    color: #f8fafc;
    font-size: 21px;
    font-weight: 800;
}

.stat-label {
    color: #8190a6;
    font-size: 10px;
    margin-top: 2px;
}

.owner-card {
    border: 1px solid rgba(129,140,248,0.18);
    border-radius: 15px;
    background: rgba(30,41,59,0.55);
    padding: 13px;
    margin: 15px 0;
}

.owner-title {
    color: #a5b4fc;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
}

.owner-name {
    color: #f8fafc;
    font-size: 14px;
    font-weight: 700;
    margin-top: 5px;
}

.owner-dob {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 4px;
}

.small {
    color: #718096;
    font-size: 11px;
    line-height: 1.6;
}

.search-result {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    background: #111827;
    padding: 14px;
    margin: 9px 0;
}

.search-title {
    font-weight: 700;
    color: #93c5fd;
}

.search-url {
    font-size: 11px;
    color: #64748b;
    word-break: break-all;
}

.search-snippet {
    color: #aab5c7;
    font-size: 13px;
    margin-top: 6px;
    line-height: 1.5;
}

footer {
    text-align: center;
    color: #64748b;
    font-size: 11px;
    padding: 25px 0;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD CHATBOT MODEL
# ============================================================

@st.cache_resource
def load_bot():

    base = Path(__file__).parent

    with open(
        base / "intents.json",
        "r",
        encoding="utf-8"
    ) as file:
        intents = json.load(file)

    with open(
        base / "words.pkl",
        "rb"
    ) as file:
        words = pickle.load(file)

    with open(
        base / "classes.pkl",
        "rb"
    ) as file:
        classes = pickle.load(file)

    model = load_model(
        base / "chatbot_model.h5",
        compile=False,
        custom_objects={
            "Dense": CompatibleDense,
            "InputLayer": CompatibleInputLayer,
            "GlorotUniform": CompatibleGlorotUniform,
        }
    )

    return intents, words, classes, model


try:

    intents, words, classes, model = load_bot()

    MODEL_READY = True
    MODEL_ERROR = ""

except Exception as error:

    intents = {}
    words = []
    classes = []
    model = None

    MODEL_READY = False
    MODEL_ERROR = str(error)


lemmatizer = WordNetLemmatizer()


# ============================================================
# NLTK
# ============================================================

def setup_nltk():

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


setup_nltk()


# ============================================================
# TOKENIZATION
# ============================================================

def clean_sentence(sentence):

    try:

        sentence_words = nltk.word_tokenize(sentence)

    except LookupError:

        sentence_words = sentence.lower().split()

    return [
        lemmatizer.lemmatize(word.lower())
        for word in sentence_words
    ]


# ============================================================
# BAG OF WORDS
# ============================================================

def bag_of_words(sentence):

    sentence_words = clean_sentence(sentence)

    bag = [0] * len(words)

    for w in sentence_words:

        for i, word in enumerate(words):

            if word == w:
                bag[i] = 1

    return np.array(
        bag,
        dtype=np.float32
    )


# ============================================================
# PREDICT INTENT
# ============================================================

def predict_intent(sentence):

    if not MODEL_READY:
        return []

    try:

        bow = bag_of_words(sentence)

        result = model.predict(
            np.array([bow]),
            verbose=0
        )[0]

    except Exception:
        return []

    threshold = 0.25

    results = [
        (i, float(score))
        for i, score in enumerate(result)
        if float(score) > threshold
    ]

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "intent": classes[i],
            "probability": score
        }
        for i, score in results
    ]


# ============================================================
# LOCAL CHATBOT RESPONSE
# ============================================================

def local_bot_response(message):

    predictions = predict_intent(message)

    if not predictions:
        return None

    tag = predictions[0]["intent"]

    for item in intents.get("intents", []):

        if item.get("tag") == tag:

            responses = item.get(
                "responses",
                []
            )

            if responses:
                return random.choice(responses)

    return None


# ============================================================
# OWNER INFORMATION
# ============================================================

def owner_response(message):

    text = message.lower().strip()

    owner_words = [
        "owner",
        "owner name",
        "who is the owner",
        "who owns you",
        "who created you",
        "creator name",
        "your owner",
    ]

    dob_words = [
        "my dob",
        "owner dob",
        "date of birth",
        "owner date of birth",
        "when was i born",
        "when was the owner born",
        "birth date",
    ]

    age_words = [
        "my age",
        "owner age",
        "how old am i",
        "how old is the owner",
        "owner is how old",
    ]

    if any(word in text for word in owner_words):

        return (
            f"👤 **My owner is {OWNER_NAME}.**"
        )

    if any(word in text for word in dob_words):

        return (
            f"🎂 **Date of Birth:** {OWNER_DOB}"
        )

    if any(word in text for word in age_words):

        today = date.today()

        age = (
            today.year
            - OWNER_BIRTH_DATE.year
            - (
                (today.month, today.day)
                <
                (
                    OWNER_BIRTH_DATE.month,
                    OWNER_BIRTH_DATE.day
                )
            )
        )

        return (
            f"🎂 **{OWNER_NAME}** is "
            f"**{age} years old**."
        )

    return None


# ============================================================
# CALCULATOR
# ============================================================

def safe_calculate(expression):

    expression = expression.replace(
        "×",
        "*"
    )

    expression = expression.replace(
        "÷",
        "/"
    )

    expression = expression.replace(
        "^",
        "**"
    )

    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
        ast.Mod: op.mod,
        ast.FloorDiv: op.floordiv,
    }

    def evaluate(node):

        if isinstance(
            node,
            ast.Expression
        ):
            return evaluate(node.body)

        if isinstance(
            node,
            ast.Constant
        ):

            if isinstance(
                node.value,
                (int, float)
            ):
                return node.value

        if isinstance(
            node,
            ast.BinOp
        ):

            if type(node.op) not in operators:
                raise ValueError

            left = evaluate(node.left)
            right = evaluate(node.right)

            if (
                isinstance(
                    node.op,
                    ast.Pow
                )
                and abs(right) > 10
            ):
                raise ValueError

            return operators[
                type(node.op)
            ](
                left,
                right
            )

        if isinstance(
            node,
            ast.UnaryOp
        ):

            if type(node.op) not in operators:
                raise ValueError

            return operators[
                type(node.op)
            ](
                evaluate(node.operand)
            )

        raise ValueError

    result = evaluate(
        ast.parse(
            expression,
            mode="eval"
        )
    )

    if (
        isinstance(result, float)
        and result.is_integer()
    ):
        return int(result)

    return round(
        result,
        10
    )


def calculator_tool(text):

    text = text.strip()

    patterns = [

        r"^(?:calculate|calc|solve)\s+(.+)$",

        r"^what is\s+"
        r"([0-9.\+\-\*\/\%\(\)\^\×\÷\s]+)"
        r"\??$",

        r"^([0-9.\+\-\*\/\%\(\)\^\×\÷\s]+)$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            re.I
        )

        if match:

            expression = match.group(1)

            if re.search(
                r"\d",
                expression
            ):

                try:

                    answer = safe_calculate(
                        expression
                    )

                    return (
                        f"🧮 **{answer}**"
                    )

                except Exception:
                    return None

    return None


# ============================================================
# UNIT CONVERTER
# ============================================================

def unit_converter(text):

    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*"
        r"([a-z°]+)\s*"
        r"(?:to|in|into|=)\s*"
        r"([a-z°]+)",
        text.lower()
    )

    if not match:
        return None

    value = float(
        match.group(1)
    )

    source = match.group(2)
    target = match.group(3)

    aliases = {

        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometres": "km",

        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",

        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",

        "millimeter": "mm",
        "millimeters": "mm",

        "kilogram": "kg",
        "kilograms": "kg",

        "gram": "g",
        "grams": "g",

        "milligram": "mg",
        "milligrams": "mg",

        "liter": "l",
        "liters": "l",
        "litre": "l",
        "litres": "l",

        "milliliter": "ml",
        "milliliters": "ml",
        "millilitre": "ml",
        "millilitres": "ml",

        "feet": "ft",
        "foot": "ft",

        "miles": "mile",

        "hours": "hour",
        "minutes": "minute",
        "days": "day",
    }

    source = aliases.get(
        source.replace("°", ""),
        source.replace("°", "")
    )

    target = aliases.get(
        target.replace("°", ""),
        target.replace("°", "")
    )

    conversions = {

        ("km", "m"):
            lambda x: x * 1000,

        ("m", "km"):
            lambda x: x / 1000,

        ("m", "cm"):
            lambda x: x * 100,

        ("cm", "m"):
            lambda x: x / 100,

        ("m", "mm"):
            lambda x: x * 1000,

        ("mm", "m"):
            lambda x: x / 1000,

        ("km", "cm"):
            lambda x: x * 100000,

        ("cm", "km"):
            lambda x: x / 100000,

        ("kg", "g"):
            lambda x: x * 1000,

        ("g", "kg"):
            lambda x: x / 1000,

        ("kg", "mg"):
            lambda x: x * 1000000,

        ("mg", "kg"):
            lambda x: x / 1000000,

        ("l", "ml"):
            lambda x: x * 1000,

        ("ml", "l"):
            lambda x: x / 1000,

        ("m", "ft"):
            lambda x: x * 3.280839895,

        ("ft", "m"):
            lambda x: x / 3.280839895,

        ("km", "mile"):
            lambda x: x * 0.621371192,

        ("mile", "km"):
            lambda x: x / 0.621371192,

        ("c", "f"):
            lambda x: x * 9 / 5 + 32,

        ("f", "c"):
            lambda x: (x - 32) * 5 / 9,

        ("hour", "minute"):
            lambda x: x * 60,

        ("minute", "hour"):
            lambda x: x / 60,

        ("day", "hour"):
            lambda x: x * 24,

        ("hour", "day"):
            lambda x: x / 24,
    }

    key = (
        source,
        target
    )

    if key not in conversions:
        return None

    result = conversions[key](value)

    if float(result).is_integer():

        formatted = str(
            int(result)
        )

    else:

        formatted = f"{result:.8g}"

    return (
        f"📏 **{value:g} {source} = "
        f"{formatted} {target}**"
    )


# ============================================================
# BMI
# ============================================================

def bmi_tool(text):

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*kg"
        r".*?"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:m|meter|meters|metre|metres)",
        text.lower()
    )

    if not match:
        return None

    weight = float(
        match.group(1)
    )

    height = float(
        match.group(2)
    )

    if height <= 0:
        return None

    bmi = weight / (
        height * height
    )

    if bmi < 18.5:

        category = "Underweight"

    elif bmi < 25:

        category = "Normal weight"

    elif bmi < 30:

        category = "Overweight"

    else:

        category = "Obesity"

    return (
        f"⚖️ **BMI: {bmi:.1f}**\n\n"
        f"Category: **{category}**"
    )


# ============================================================
# AGE CALCULATOR
# ============================================================

def age_tool(text):

    lower = text.lower()

    # Owner's age
    if any(
        x in lower
        for x in [
            "my age",
            "owner age",
            "how old am i",
            "how old is the owner",
        ]
    ):

        today = date.today()

        age = (
            today.year
            - OWNER_BIRTH_DATE.year
            - (
                (today.month, today.day)
                <
                (
                    OWNER_BIRTH_DATE.month,
                    OWNER_BIRTH_DATE.day
                )
            )
        )

        return (
            f"🎂 **{OWNER_NAME}** is "
            f"**{age} years old**."
        )

    match = re.search(
        r"(?:age|born|dob).*?"
        r"(\d{1,2})"
        r"[\/\-.]"
        r"(\d{1,2})"
        r"[\/\-.]"
        r"(\d{4})",
        lower
    )

    if not match:
        return None

    day = int(
        match.group(1)
    )

    month = int(
        match.group(2)
    )

    year = int(
        match.group(3)
    )

    try:

        birthday = date(
            year,
            month,
            day
        )

    except ValueError:

        return (
            "❌ Please enter a valid date."
        )

    today = date.today()

    age = (
        today.year
        - birthday.year
        - (
            (today.month, today.day)
            <
            (
                birthday.month,
                birthday.day
            )
        )
    )

    return (
        f"🎂 Age: **{age} years**"
    )


# ============================================================
# TIME AND DATE
# ============================================================

def time_date_tool(text):

    lower = text.lower()

    time_words = [
        "time",
        "current time",
        "what time",
        "live time",
    ]

    date_words = [
        "date",
        "today",
        "today's date",
        "what day",
    ]

    wants_time = any(
        x in lower
        for x in time_words
    )

    wants_date = any(
        x in lower
        for x in date_words
    )

    if not wants_time and not wants_date:
        return None

    now = datetime.now()

    if wants_time and wants_date:

        return (
            f"🕒 **{now.strftime('%I:%M:%S %p')}**\n\n"
            f"📅 **{now.strftime('%A, %d %B %Y')}**"
        )

    if wants_time:

        return (
            f"🕒 Current time: "
            f"**{now.strftime('%I:%M:%S %p')}**"
        )

    return (
        f"📅 Today is "
        f"**{now.strftime('%A, %d %B %Y')}**"
    )


# ============================================================
# JOKES
# ============================================================

def joke_tool(text):

    if not any(
        x in text.lower()
        for x in [
            "joke",
            "funny",
            "make me laugh"
        ]
    ):
        return None

    jokes = [

        "😂 Why do programmers prefer dark mode? "
        "Because light attracts bugs!",

        "🤣 Why did the programmer quit? "
        "Because they didn't get arrays!",

        "😄 A SQL query walks into a bar "
        "and asks two tables: Can I join you?",

        "😂 I told my computer I needed a break. "
        "Now it keeps showing me KitKat ads.",

        "😎 Why was Python calm? "
        "It knew how to handle its exceptions.",
    ]

    return random.choice(jokes)


# ============================================================
# WEATHER
# ============================================================

@st.cache_data(ttl=600)
def weather_lookup(city):

    try:

        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=8
        )

        geo.raise_for_status()

        results = geo.json().get(
            "results",
            []
        )

        if not results:

            return (
                "❌ City not found."
            )

        place = results[0]

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":
                    place["latitude"],

                "longitude":
                    place["longitude"],

                "current":
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "wind_speed_10m,"
                    "weather_code",

                "timezone": "auto",
            },
            timeout=8
        )

        response.raise_for_status()

        current = response.json()[
            "current"
        ]

        conditions = {

            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",

            45: "Fog",
            48: "Rime fog",

            51: "Light drizzle",
            53: "Drizzle",
            55: "Heavy drizzle",

            61: "Light rain",
            63: "Rain",
            65: "Heavy rain",

            71: "Light snow",
            73: "Snow",
            75: "Heavy snow",

            80: "Rain showers",
            81: "Rain showers",
            82: "Heavy rain showers",

            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with hail",
        }

        condition = conditions.get(
            current["weather_code"],
            "Unknown"
        )

        return (
            f"🌤️ **Weather in {place['name']}**\n\n"
            f"🌡️ Temperature: "
            f"**{current['temperature_2m']} °C**\n\n"
            f"☁️ Condition: **{condition}**\n\n"
            f"💧 Humidity: "
            f"**{current['relative_humidity_2m']}%**\n\n"
            f"💨 Wind: "
            f"**{current['wind_speed_10m']} km/h**"
        )

    except Exception:

        return (
            "⚠️ Weather service is "
            "currently unavailable."
        )


# ============================================================
# CURRENCY
# ============================================================

@st.cache_data(ttl=600)
def convert_currency(
    amount,
    source,
    target
):

    response = requests.get(
        "https://api.frankfurter.app/latest",
        params={
            "amount": amount,
            "from": source,
            "to": target,
        },
        timeout=8
    )

    response.raise_for_status()

    return float(
        response.json()["rates"][target]
    )


def currency_tool(text):

    match = re.search(
        r"(?:convert\s*)?"
        r"(\d+(?:\.\d+)?)\s*"
        r"([A-Za-z]{3})\s*"
        r"(?:to|in|into)\s*"
        r"([A-Za-z]{3})",
        text,
        re.I
    )

    if not match:
        return None

    amount = float(
        match.group(1)
    )

    source = match.group(2).upper()
    target = match.group(3).upper()

    supported = {
        "USD",
        "INR",
        "EUR",
        "GBP",
        "JPY",
        "AUD",
        "CAD",
        "SGD",
        "AED",
        "SAR",
        "CHF",
        "CNY",
        "NZD",
        "HKD",
    }

    if (
        source not in supported
        or target not in supported
    ):
        return None

    try:

        result = convert_currency(
            amount,
            source,
            target
        )

        return (
            f"💱 **{amount:g} {source} = "
            f"{result:,.2f} {target}**"
        )

    except Exception:

        return (
            "⚠️ Currency service is "
            "currently unavailable."
        )


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query):

    try:

        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 0,
            },
            timeout=8
        )

        response.raise_for_status()

        data = response.json()

        results = []

        if (
            data.get("AbstractText")
            and data.get("AbstractURL")
        ):

            results.append({
                "title":
                    data.get(
                        "Heading",
                        query
                    ),

                "url":
                    data["AbstractURL"],

                "snippet":
                    data["AbstractText"],
            })

        def collect(items):

            for item in items:

                if len(results) >= 6:
                    return

                if (
                    isinstance(item, dict)
                    and item.get("FirstURL")
                ):

                    results.append({
                        "title":
                            item.get(
                                "Text",
                                "Search result"
                            ),

                        "url":
                            item["FirstURL"],

                        "snippet":
                            item.get(
                                "Text",
                                ""
                            ),
                    })

                if (
                    isinstance(item, dict)
                    and item.get("Topics")
                ):

                    collect(
                        item["Topics"]
                    )

        collect(
            data.get(
                "RelatedTopics",
                []
            )
        )

        return results[:6]

    except Exception:

        return []


def search_response(query):

    results = web_search(query)

    if not results:

        google_url = (
            "https://www.google.com/search?q="
            + quote_plus(query)
        )

        return (
            "🔎 Search preview unavailable.\n\n"
            f"[Open Google Search]({google_url})"
        )

    output = [
        f"### 🔎 Search results for **{query}**"
    ]

    for result in results:

        output.append(
            f"**[{result['title']}]"
            f"({result['url']})**\n\n"
            f"{result['snippet']}"
        )

    return "\n\n---\n\n".join(output)


# ============================================================
# SMART RESPONSE
# ============================================================

def smart_response(message):

    text = message.strip()
    lower = text.lower()

    # --------------------------------------------------------
    # OWNER
    # --------------------------------------------------------

    result = owner_response(text)

    if result:
        return result

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_match = re.match(
        r"(?:search|google|web search|"
        r"search web|look up)\s+"
        r"(?:for\s+)?(.+)",
        text,
        re.I
    )

    if search_match:

        query = search_match.group(1).strip()

        return search_response(query)

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    weather_match = re.search(
        r"(?:weather|temperature|forecast)"
        r"\s+(?:in|at|for)\s+"
        r"([A-Za-z .'-]+)",
        text,
        re.I
    )

    if weather_match:

        city = weather_match.group(1).strip(
            " .?!"
        )

        return weather_lookup(city)

    # --------------------------------------------------------
    # UNIT CONVERSION
    # --------------------------------------------------------

    result = unit_converter(text)

    if result:
        return result

    # --------------------------------------------------------
    # CURRENCY
    # --------------------------------------------------------

    result = currency_tool(text)

    if result:
        return result

    # --------------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------------

    result = calculator_tool(text)

    if result:
        return result

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    result = bmi_tool(text)

    if result:
        return result

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    result = age_tool(text)

    if result:
        return result

    # --------------------------------------------------------
    # TIME / DATE
    # --------------------------------------------------------

    result = time_date_tool(text)

    if result:
        return result

    # --------------------------------------------------------
    # JOKE
    # --------------------------------------------------------

    result = joke_tool(text)

    if result:
        return result

    # --------------------------------------------------------
    # LOCAL TRAINED MODEL
    # --------------------------------------------------------

    result = local_bot_response(text)

    if result:
        return result

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if any(
        x in lower
        for x in [
            "who are you",
            "what are you",
            "your name",
        ]
    ):

        return (
            "🤖 I'm **GO! AI**, your smart "
            "personal assistant.\n\n"
            "I can chat, calculate, convert "
            "units, check weather, convert "
            "currency, calculate BMI/age, "
            "tell jokes and search the web."
        )

    if (
        "help" in lower
        or "what can you do" in lower
    ):

        return (
            "### 🤖 I can help with\n\n"
            "💬 Chat\n\n"
            "🌤️ Weather\n\n"
            "💱 Currency\n\n"
            "🧮 Calculator\n\n"
            "📏 Unit conversion\n\n"
            "⚖️ BMI\n\n"
            "🎂 Age\n\n"
            "🕒 Time & date\n\n"
            "😂 Jokes\n\n"
            "🔎 Web search\n\n"
            "**Examples:**\n\n"
            "`10 km to m`\n\n"
            "`100 USD to INR`\n\n"
            "`weather in Delhi`\n\n"
            "`25 * 8`\n\n"
            "`tell me a joke`\n\n"
            "`who is the owner`"
        )

    return (
        "🤔 I didn't fully understand that.\n\n"
        "Try **help** to see what I can do."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
        🤖 GO! AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-sub">
        Smart Personal Assistant
        </div>
        """,
        unsafe_allow_html=True
    )

    # Owner card

    st.markdown(
        f"""
        <div class="owner-card">

        <div class="owner-title">
        👤 App Owner
        </div>

        <div class="owner-name">
        {OWNER_NAME}
        </div>

        <div class="owner-dob">
        🎂 DOB: {OWNER_DOB}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # New chat

    if st.button(
        "✨ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown(
        "### ⚡ Quick Tools"
    )

    quick_tools = [

        (
            "🌤️",
            "Weather",
            "weather in Delhi"
        ),

        (
            "💱",
            "Currency",
            "100 USD to INR"
        ),

        (
            "🧮",
            "Calculator",
            "125 * 8 + 20"
        ),

        (
            "📏",
            "Units",
            "10 km to m"
        ),

        (
            "⚖️",
            "BMI",
            "BMI 70 kg 1.75 m"
        ),

        (
            "🎂",
            "My Age",
            "my age"
        ),

        (
            "🕒",
            "Time",
            "what is the time"
        ),

        (
            "😂",
            "Joke",
            "tell me a joke"
        ),

        (
            "🔎",
            "Search",
            "search Python AI"
        ),
    ]

    for icon, label, prompt in quick_tools:

        if st.button(
            f"{icon} {label}",
            key=f"quick_{label}",
            use_container_width=True
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

            with st.spinner(
                "Thinking..."
            ):

                answer = smart_response(
                    prompt
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.rerun()

    st.markdown(
        "### 📊 Conversation"
    )

    user_messages = sum(
        x["role"] == "user"
        for x in st.session_state.messages
    )

    ai_messages = sum(
        x["role"] == "assistant"
        for x in st.session_state.messages
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="stat">

            <div class="stat-number">
            {user_messages}
            </div>

            <div class="stat-label">
            YOU
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="stat">

            <div class="stat-number">
            {ai_messages}
            </div>

            <div class="stat-label">
            AI
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "### 💾 Download"
    )

    if st.session_state.messages:

        conversation = (
            "GO! AI CONVERSATION\n"
            "====================\n\n"
        )

        for message in st.session_state.messages:

            who = (
                "YOU"
                if message["role"] == "user"
                else "GO! AI"
            )

            conversation += (
                f"{who}:\n"
                f"{message['content']}\n\n"
            )

        st.download_button(
            "📄 Download TXT",
            conversation,
            file_name=(
                f"go_ai_chat_"
                f"{datetime.now():%Y%m%d_%H%M%S}"
                f".txt"
            ),
            mime="text/plain",
            use_container_width=True
        )

        st.download_button(
            "🧾 Download JSON",
            json.dumps(
                st.session_state.messages,
                ensure_ascii=False,
                indent=2
            ),
            file_name=(
                f"go_ai_chat_"
                f"{datetime.now():%Y%m%d_%H%M%S}"
                f".json"
            ),
            mime="application/json",
            use_container_width=True
        )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown(
        "### ℹ️ System"
    )

    if MODEL_READY:

        st.success(
            "Local chatbot model loaded"
        )

    else:

        st.error(
            "Model files not found"
        )

        st.caption(
            MODEL_ERROR
        )


# ============================================================
# MAIN HEADER
# ============================================================

status_text = (
    "Local AI model ready"
    if MODEL_READY
    else
    "Local AI model unavailable"
)

st.markdown(
    f"""
    <div class="hero">

    <div class="hero-title">
    🤖 GO! AI
    </div>

    <div class="hero-subtitle">
    Your modern Python AI assistant with
    smart tools, web search, live information
    and conversation export.
    </div>

    <div class="status">
    <span class="dot"></span>
    {status_text}
    </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:30px 10px 25px;
        ">

        <div style="font-size:50px;">
        ✨
        </div>

        <h2 style="
            color:#f8fafc;
            margin:8px 0;
        ">
        How can I help you today?
        </h2>

        <p style="
            color:#8190a6;
        ">
        Ask anything supported by your
        chatbot and smart tools.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    cards = [

        (
            "🧮",
            "Calculator",
            "25 * 8 + 10"
        ),

        (
            "📏",
            "Unit Conversion",
            "10 km to m"
        ),

        (
            "🌤️",
            "Weather",
            "weather in Delhi"
        ),

        (
            "🔎",
            "Web Search",
            "search Python AI"
        ),
    ]

    columns = st.columns(4)

    for column, card in zip(
        columns,
        cards
    ):

        icon, name, example = card

        with column:

            st.markdown(
                f"""
                <div class="tool-card">

                <div class="tool-icon">
                {icon}
                </div>

                <div class="tool-name">
                {name}
                </div>

                <div class="tool-example">
                {example}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# DISPLAY CHAT
# ============================================================

for message in st.session_state.messages:

    avatar = (
        "👤"
        if message["role"] == "user"
        else
        "🤖"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Message GO! AI..."
)


if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(prompt)

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "Thinking..."
        ):

            answer = smart_response(
                prompt
            )

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <footer>
    GO! AI · Owner: {OWNER_NAME}
    · DOB: {OWNER_DOB}
    · Python · TensorFlow · Keras
    · NLTK · Streamlit
    </footer>
    """,
    unsafe_allow_html=True
)
