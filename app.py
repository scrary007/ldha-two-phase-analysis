"""LDHA biphasic NADH time-course analysis: standalone Streamlit dashboard.

Usage: streamlit run app.py
No cloud database, file storage, or Google credentials are required.
"""
from __future__ import annotations

from hashlib import sha1
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analysis import (
    DataError, filename_pyruvate, fit_all_models, fit_line,
    fit_substrate_inhibition, read_cary_csv, recommended_windows,
    signal_slope_to_velocity, substrate_inhibition,
)
from report import make_pdf


st.set_page_config(page_title='LDHA | Biphasic time courses', layout='wide',
                   page_icon='🧪')
BLUE, ORANGE, GREEN, PURPLE, GRAY = '#0072B2','#D55E00','#009E73','#CC79A7','#4A4A4A'

st.title('LDHA · two-phase time-course analyzer')
st.caption('Exploratory fitting of NADH absorbance time courses at 340 nm')
st.info('This app describes early and late reaction rates. A two-phase trace **does not by itself** '
        'establish formation of a pyruvate–NAD⁺ adduct. Compare independent experiments '
        'and rule out mixing, NADH depletion, and product accumulation.')

with st.sidebar:
    st.header('1 · Upload Cary files')
    uploaded = st.file_uploader('Time-course CSV files', type=['csv','txt'],
                                accept_multiple_files=True,
                                help='Select one or many raw A340 files.')
    st.header('2 · Instrument and conversion')
    path_cm = st.number_input('Cuvette optical path length (cm)', min_value=.01,
                              max_value=10.0, value=1.0, step=.1, format='%.2f')
    extinction = st.number_input('NADH ε340 (mM⁻¹ cm⁻¹)', min_value=.01,
                                 max_value=100.0, value=6.22, step=.01, format='%.2f')
    initial_nadh = st.number_input('Initial NADH (μM)', min_value=.01,
                                   value=75.0, step=5.0)
    blank_slope = st.number_input('Optional blank A340 drift (A/s)',
                                  value=0.0, format='%.7f',
                                  help='Subtract this measured blank slope from all fitted slopes.')
    st.header('3 · CSV parsing')
    with st.expander('Advanced: columns and units'):
        t_col = st.number_input('Time column (1 = first)', min_value=1, value=1)
        a_col = st.number_input('Absorbance column (1 = first)', min_value=1, value=2)
        time_unit = st.selectbox('Cary time units', ['seconds','minutes'], index=0)
    st.header('4 · Fit range')
    dead_time_s = st.number_input('Exclude data before this time (s)',
                                  min_value=0.0, value=5.0, step=1.0)

st.markdown('**Models:** two selected linear windows · continuous two-slope breakpoint · '
            'smooth exponential slope transition · single straight-line reference.')

if not uploaded:
    st.subheader('Start with a few traces')
    st.markdown('Upload several Cary CSV files. The app tries to read [pyruvate] from filenames '
                'such as `RS_WT_53,3mM_A.csv`; you can edit concentrations in the table. '
                'For a first check, use low- and high-pyruvate traces from the same enzyme.')
    st.markdown('The downloaded project contains **10 synthetic example CSVs** in `sample_data/` '
                'for testing the interface. These are not experimental measurements.')
    st.stop()

if len({file.name for file in uploaded}) != len(uploaded):
    st.error('Two uploaded files have the same filename. Rename one before uploading.')
    st.stop()

parsed = {}
problems = []
for file in uploaded:
    blob = file.getvalue()
    uid = sha1(blob + file.name.encode()).hexdigest()[:12]
    try:
        df = read_cary_csv(blob, time_column=int(t_col),
                           absorbance_column=int(a_col), time_unit=time_unit)
        parsed[file.name] = {'df': df, 'uid': uid}
    except (DataError, UnicodeError) as e:
        problems.append(f'{file.name}: {e}')
for issue in problems:
    st.warning(issue)
if not parsed:
    st.error('No valid time-course files could be read.')
    st.stop()

st.subheader('Select experiments and enter concentrations')
meta = pd.DataFrame([{'Filename': name, '[Pyruvate] (mM)': filename_pyruvate(name),
                      'Enzyme': 'WT', 'Include': True} for name in parsed])

