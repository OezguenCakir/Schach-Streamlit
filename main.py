import pandas as pd
import urllib
import urllib.request
import requests
import json
from datetime import date, datetime
import time
import numpy as np
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb


if "button_clicked" not in st.session_state:    
    st.session_state.button_clicked = False

def callback():
    # Button wurde geklickt
    st.session_state.button_clicked = True


@st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
def datenziehung(username):
    # Use of the API to download monthly archives
    baseUrl = "https://api.chess.com/pub/player/" + username + "/games/"

    # read the archives url and store in a list
    f = urllib.request.urlopen(baseUrl + "archives")
    archives = f.read().decode("utf-8")
    archives = archives.replace("{\"archives\":[\"", "\",\"")
    archivesList = archives.split("\",\"" + baseUrl)
    archivesList[len(archivesList)-1] = archivesList[len(archivesList)-1].rstrip("\"]}")
    df_list = []
    p = st.empty()
    progressbar = st.progress(0)
    for i in range(len(archivesList)-1):
            url = baseUrl + archivesList[i+1]
            # Make an HTTP request to the URL
            response = requests.get(url)
            # Parse the response into a Python dictionary
            data = json.loads(response.text)
            # Convert the dictionary into a list of records
            records = data['games']
            # Convert the list of records into a dataframe and append it to the list
            df_list.append(pd.DataFrame.from_records(records))
            p.caption(url + " wird gespeichert")
            i = i+ 1
            progressbar.progress( (i+1)/len(archivesList) )
    # Concatenate all the dataframes into a single dataframe
    df = pd.concat(df_list) 
    return df

@st.cache(suppress_st_warning=True, show_spinner=False)
def datenbearbeitung(df):
    # Spalten-Titel umbennen
    dict = {'url': 'Link',
            'pgn': 'PGN',
            'time_control': 'TimeControl',
            'end_time': 'EndTime',
            'rated':'Bewertet',
            'start_time':'StartTime'
            }

    df.rename(columns=dict,
            inplace=True)

    df['meine Farbe'] = np.where(
        df['white'].apply(lambda x: x.get('username')) == text_input, 
        'Weiß', 
        'Schwarz'
        )

    df['mein Elo'] = np.where(
        df['meine Farbe'] == 'Weiß',
        df['white'].apply(lambda x: x.get('rating')), 
        df['black'].apply(lambda x: x.get('rating'))
        )

    df['Gegner Elo'] = np.where(
        df['meine Farbe'] == 'Weiß',
        df['black'].apply(lambda x: x.get('rating')), 
        df['white'].apply(lambda x: x.get('rating'))
        )

    df['Ergebnis'] = np.where(
        df['meine Farbe'] == 'Weiß',
        df['white'].apply(lambda x: x.get('result')),
        df['black'].apply(lambda x: x.get('result'))
    )

    df['Gegner Ergebnis'] = np.where(
        df['meine Farbe'] == 'Weiß',
        df['black'].apply(lambda x: x.get('result')),
        df['white'].apply(lambda x: x.get('result'))
    )

    df['Weiß Ergebnis'] = df['white'].apply(lambda x: x.get('result')) #drinlassen für Kontrolle

    df['Schwarz Ergebnis'] = df['black'].apply(lambda x: x.get('result')) #drinlassen für Kontrolle

    df['Gegner-Name'] = np.where(
        df['meine Farbe'] == 'Weiß',
        df['black'].apply(lambda x: x.get('username')),
        df['white'].apply(lambda x: x.get('username'))
    )

    df['Gegner API'] = np.where(
        df['meine Farbe'] == 'Weiß',
        df['black'].apply(lambda x: x.get('@id')),
        df['white'].apply(lambda x: x.get('@id'))
    )

    game_type_mapping = {
        "rapid":"Schnellschach",
        "bullet":"Bullet",
        "blitz":"Blitz",
        "daily":"Täglich",
        
    }
    
    df['Spiel-Art'] = df['time_class'].astype(str).map(game_type_mapping).astype('category')

    time_control_mapping = {
        "600":"10 Minuten",
        "60":"1 Minute",
        "60+1":"1 | 1",
        "300":"5 Minuten",
        "1/172800":"2 Tage",
        "180":"3 Minuten",
        "1800":"30 Minuten",
        "120+1":"2 | 1",
        "900+5":"10 | 5",
        "900":"15 Minuten",
        "300+5":"5 | 5",
        "180+2":"3 | 2",
        "30":"30 Sek.",
        "900+10":"10 | 10",
        "1/86400":"1 Tag"
    }	
    
    df['Zeit']= df['TimeControl'].astype(str).map(time_control_mapping)
    import datetime
    df['End-Datum'] = df['EndTime'].apply(datetime.datetime.fromtimestamp)

    df['Datum'] = df['End-Datum'].apply(lambda x: x.date())

    df['Uhrzeit'] = pd.to_datetime(df['End-Datum'], format="%H:%M:%S").dt.strftime("%H:%M:%S")
    #df['End-Datum'].apply(lambda x: x.time())
    
    if 'accuracies' not in df.columns:
        df['accuracies']= None
        df['accuracies'] = np.where(
            df.accuracies.isna(),
            {},
            df.accuracies
        ) 

    df['Meine Genauigkeit'] = np.where(
        df.accuracies.isna(),
        {},
        df.accuracies
    )

    df['Meine Genauigkeit'] = np.where(
        df['meine Farbe'] == 'Weiß',
        df['Meine Genauigkeit'].apply(lambda x: x.get('white')),
        df['Meine Genauigkeit'].apply(lambda x: x.get('black'))   
    )

    df['Meine Genauigkeit'] =["{:.1%}".format(x/100) for x in df['Meine Genauigkeit']]

    df['Gegner Genauigkeit'] = np.where(
        df.accuracies.isna(),
        {},
        df.accuracies
    )

    df['Gegner Genauigkeit'] = np.where(
        df['meine Farbe'] == 'Weiß',
        df['Gegner Genauigkeit'].apply(lambda x: x.get('black')),
        df['Gegner Genauigkeit'].apply(lambda x: x.get('white'))   
    )
    
    df['Gegner Genauigkeit'] =["{:.1%}".format(x/100) for x in df['Gegner Genauigkeit']]

    ergebnis_mapping = {
        "checkmated":"Schachmatt",
        "timeout":"Zeitüberschreitung",
        "resigned":"Aufgabe",
        "win":"gewonnen",
        "stalemate":"Patt",
        "abandoned":"Verlassen",
        "repetition":"Stellungswiederholung",
        "timevsinsufficient":"Zeitüberschreitung gegen unzureichendes Material",
        "threecheck":"3 Mal Schach-gestellt",
        "agreed":"Vereinbarung",
        "insufficient":"Nicht genügend Material",
        "50move":"50 Züge Regel",
        "lose":"verloren",
        "kingofthehill":"Gegnerischer König hat die Mitte erreicht",
        "bughousepartnerlose":"Bughouse-Partner hat verloren"
    }	

    df['Ausgang'] = np.where(
        df['Ergebnis'] == 'win',
        'gewonnen',
        np.where(
            df['Gegner Ergebnis'] == 'win',
            'verloren',
            'unentschieden'
        )
    )

    df['Ausgang-Grund'] = np.where(
        df['Ergebnis'] == 'win',
        df['Gegner Ergebnis'].astype(str).map(ergebnis_mapping),
        df['Ergebnis'].astype(str).map(ergebnis_mapping)
    )

    ergebnis_mapping = {
        "chess":"klassisch",
        "kingofthehill":"King of the Hill",
        "threecheck":"3 Schach Schach",
        "chess960":"Schach960",
        "crazyhouse":"Einsetzschach",
        "bughouse":"Tandemschach",
        "oddschess":"Odds Chess"
    }	
    
    df['Variante'] = df.rules.map(ergebnis_mapping)

    time_control_mapping = {
        "10":"10 Sek.",
        "30":"30 Sek.",
        "60":"1 Minute",
        "60+1":"1 | 1",
        "120+1":"2 | 1",

        "180":"3 Minuten",
        "180+1":"3 | 1",
        "180+2":"3 | 2",
        "180+3":"3 | 3",
        "180+5":"3 | 5",
        "300":"5 Minuten",
        "300+1":"5 | 1",
        "300+2":"5 | 2",
        "300+3":"5 | 3",
        "300+5":"5 | 5",
        "300+10":"5 | 10",
        "420":"7 Minuten",
        "480":"8 Minuten",
        "480+2":"8 | 2",

        "600":"10 Minuten",
        "600+2":"10 | 2",
        "600+3":"10 | 3",
        "600+5":"10 | 5",
        "600+10":"10 | 10",
        "900":"15 Minuten",
        "900+2":"15 | 2",
        "900+3":"15 | 3",
        "900+5":"15 | 5",
        "900+10":"15 | 10",
        "1200":"20 Minuten",
        "1200+10":"20 | 10",
        "1800":"30 Minuten",
        "2700+15":"45 | 15",
        "3600+45":"60 | 45",

        "1/86400":"1 Tag",
        "1/172800":"2 Tage",
        "1/259200":"3 Tage",
        "1/432000":"5 Tage",
        "1/604800":"7 Tage",
        "1/1209600":"14 Tage"
    }	
    
    df['Zeit']= df['TimeControl'].astype(str).map(time_control_mapping)

    wochentag_mapping = {
        0:"Montag",
        1:"Dienstag",
        2:"Mittwoch",
        3:"Donnerstag",
        4:"Freitag",
        5:"Samstag",
        6:"Sonntag",
        7:"2 | 1"
    }	
    
    df['Wochentag']= df['End-Datum'].apply(lambda time: time.dayofweek).map(wochentag_mapping)


    df = df[[
        'Datum','Uhrzeit','Wochentag', 'Spiel-Art', 'Zeit', 'Variante', 'Bewertet',
        'Ausgang', 'Ausgang-Grund',
        'meine Farbe', 'mein Elo', 'Gegner Elo', 'Gegner-Name',
        'Meine Genauigkeit', 'Gegner Genauigkeit', 'Link', 'Anzahl Züge', 'Rest-Zeit'
    ]]

    df = df.sort_values('Datum', ascending=False).reset_index(drop=True)
    df.index = np.arange(1, len(df) + 1)
    

    return df

def df_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name= text_input + ' Chess.com Data')
    workbook = writer.book
    workbook.formats[0].set_font_name('Helvetica')
    workbook.formats[0].set_border(2)
    worksheet = writer.sheets[text_input + ' Chess.com Data']
    worksheet.autofilter(0, 0, df.shape[0], df.shape[1]-1) # activate filter on column headers
    header_format = workbook.add_format({ # define header format
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#006E9D',
            'font_color': 'white',
            'border': 0,
            'font_name': 'Helvetica'
            })
    for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            max_len = max(
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
                ) + 5  # adding more space
            worksheet.set_column(idx, idx, max_len)  # set column width
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format) # set header format
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def function_erstes():
    erstes_datum = min(df['Datum'])     #pd.to_datetime( min(df['Datum']))
    return erstes_datum

def function_letztes():
    letztes_datum = max(df['Datum'])    #pd.to_datetime( max(df['Datum']) )
    return letztes_datum

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Füge Filter hinzu")
    if not modify:
        return df
    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter Datensatz mit", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Werte für {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Werte für {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Werte für {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_cat_input = right.multiselect(
                    f"Werte für {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
    return df



st.title('♟️ Chess.com Profil-Auswertung')


with st.form(key='my_form'):
	text_input = st.text_input(label='**Gebe deinen Chess.com Usernamen ein**')
	submit_button = st.form_submit_button(label='anfangen', on_click=callback)
if submit_button or st.session_state.button_clicked:
    username = text_input.lower()

    st.markdown('**Daten aus Chess.com werden gespeichert**')  
    df = datenziehung(username)
           
    st.success ("Alle Dateien wurden gespeichert ✅")
    df = datenbearbeitung(df)
    st.text('')   
else:
    st.subheader('Oder schaue dir meine Daten an')
    username = 'oezguen'
    text_input = 'oezguen'
    df = pd.read_csv('meine_daten.csv')
    df.index = np.arange(1, len(df) + 1)
    df = df[[
        'Datum','Uhrzeit','Wochentag', 'Spiel-Art', 'Zeit', 'Variante', 'Bewertet',
        'Ausgang', 'Ausgang-Grund',
        'meine Farbe', 'mein Elo', 'Gegner Elo', 'Gegner-Name',
        'Meine Genauigkeit', 'Gegner Genauigkeit', 'Link'
    ]]


with st.sidebar:
    st.subheader('Filter')

    radio_spielart = st.radio(
        "Spiel-Art",
        ("nur klassisches Schach", "alle Varianten"),
        help='es gibt besondere Varianten mit unterschiedlichen Zielen. Dieser Filter blendet diese aus'
    )
    if radio_spielart == 'nur klassisches Schach':
        df = df[ (df.Variante=='klassisch') ]
    else:
        df = df
    
    radio_farbe = st.radio(
        "Meine Farbe",
        ("Alle", "Weiß", "Schwarz")
    )
    if radio_farbe == 'Alle':
        df = df
    elif radio_farbe == 'Schwarz':
        df = df[ (df['meine Farbe']=='Schwarz') ]
    elif radio_farbe == 'Weiß':
        df = df[ (df['meine Farbe']=='Weiß') ]

    radio_ausgang = st.radio(
        "Ausgang",
        ("Alle", "gewonnen", "unentschieden", "verloren")
    )
    if radio_ausgang == 'gewonnen':
        df = df[ (df['Ausgang']=='gewonnen') ]
    elif radio_ausgang == 'unentschieden':
        df = df[ (df['Ausgang']=='unentschieden') ]
    elif radio_ausgang == 'verloren':
        df = df[ (df['Ausgang']=='verloren') ]
    elif radio_ausgang == 'Alle':
        df = df

    a_date = st.date_input(
            label= "Datum",
            min_value = pd.to_datetime( min(df['Datum']) ),
            max_value = pd.to_datetime( max(df['Datum']) ),
            value = (pd.to_datetime(min(df['Datum'])), pd.to_datetime(max(df['Datum'])) )
        )
    df = df[ ( pd.to_datetime(df['Datum']) >= pd.to_datetime(a_date[0]) ) & (pd.to_datetime(df['Datum']) <= pd.to_datetime(a_date[1]) ) ]

    st.subheader('Downloade deine Daten')

    st.download_button(
        label='📥 Excel-Datei',
        data=df_to_excel(df) ,
        file_name= 'chess_com_daten.xlsx')

    st.download_button(
        "📥 CSV-Datei",
        df_to_csv(df),
        "chess_com_daten.csv",
        "text/csv"
        )


url = "https://api.chess.com/pub/player/" + username
data = json.loads(urllib.request.urlopen(url).read())
col1, col2 = st.columns([1,5])
if data.get('avatar') == None:
    profile_pic = 'https://www.chess.com/bundles/web/images/user-image.007dad08.svg'
else:
    profile_pic = data.get('avatar')

joined_delta = date.today() - date.fromtimestamp(data.get('joined'))
last_online_delta = date.today() - date.fromtimestamp(data.get('last_online'))

col1.image(profile_pic, width=100)
col2.write("[" + data.get('username') + "](" + data.get('url') + ")  \n" + data.get('name'))
col2.caption("Follower: " + str(data.get('followers')))
col2.caption(
    "Zuletzt online am " + time.strftime('%d.%m.%Y um %H:%M:%S', time.localtime(data.get('last_online'))) + 
     " *- vor *" + str(round(last_online_delta.days,1)).replace(".",",") + '* Tagen*' + "  \n" + 
    "Registriert am " + time.strftime('%d.%m.%Y um %H:%M:%S', time.localtime(data.get('joined'))) +
    " *- vor *" + str(round(joined_delta.days/365.25,1)).replace(".",",") + '* Jahren*'
    )


st.header("Der Datensatz")
st.dataframe(filter_dataframe(df.drop(columns=['Meine Genauigkeit', 'Gegner Genauigkeit'])))






st.header('Ausgang der Spiele')
col1, col2, col3, col4 = st.columns(4)
col1.metric( 'Anzahl Spiele insgesamt', len(df.index) )
col2.metric( 'gewonnen', "{:.0%}".format(len(df.loc[(df.Ausgang == "gewonnen")]) / len(df.index)) )
col3.metric( 'unentschieden', "{:.0%}".format(len(df.loc[(df.Ausgang == "unentschieden")]) / len(df.index)) )
col4.metric( 'verloren', "{:.0%}".format(len(df.loc[(df.Ausgang == "verloren")]) / len(df.index)) )

fig = px.sunburst(
    df, 
    path=['Ausgang', 'Ausgang-Grund'],
    color='Ausgang',
    color_discrete_map={
        "gewonnen": "#007B57",
        "unentschieden": "grey",
        "verloren": "#9d2e00"
        }
    )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_traces(hovertemplate="%{label}: %{value} Spiele<extra></extra>")
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})

st.caption('Klick in den inneren Kreis und sieh was passiert')



st.subheader('Elo nach Spielen')
st.caption('Elo ist eine Kennzahl zur Bewertung der Spielstärke')

radio_spielart = st.radio(
        "Spiel-Art",
        ("Bullet", "Blitz", "Schnellschach", "Täglich"),
        horizontal=True,
        help='Dieser Filter wirkt nur auf diese Visualisierung ein',
        index=2
    )
if radio_spielart == 'Bullet':
    df_elo = df[ (df['Spiel-Art']=='Bullet') ].sort_values('Datum').reset_index(drop=True)
elif radio_spielart == 'Blitz':
    df_elo = df[ (df['Spiel-Art']=='Blitz') ].sort_values('Datum').reset_index(drop=True)
elif radio_spielart == 'Schnellschach':
    df_elo = df[ (df['Spiel-Art']=='Schnellschach') ].sort_values('Datum').reset_index(drop=True)
elif radio_spielart == 'Täglich':
    df_elo = df[ (df['Spiel-Art']=='Täglich') ].sort_values('Datum').reset_index(drop=True)

try: 
    letzter_elo = df_elo['mein Elo'].iloc[-1]
except Exception:
    letzter_elo = np.nan

try:
    elo_delta = "{:.0%}".format( (df_elo['mein Elo'].iloc[-1] - df_elo['mein Elo'].iloc[-15]) / df_elo['mein Elo'].iloc[-15] )
except Exception:
    elo_delta = np.nan

try:
    max_elo = df_elo['mein Elo'].max()
except Exception:
    max_elo = np.nan

try:
    max_delta = "{:.0%}".format( (df_elo['mein Elo'].iloc[-1] - df_elo['mein Elo'].max()) / df_elo['mein Elo'].max() )
except Exception:
    max_delta = np.nan

try:
    min_elo = df_elo['mein Elo'].min()
except Exception:
    min_elo = np.nan

try:
    min_delta = "{:.0%}".format( (df_elo['mein Elo'].iloc[-1] - df_elo['mein Elo'].min()) / df_elo['mein Elo'].min() )
except Exception:
    min_delta = np.nan

try:
    avg_elo = int(round(sum(df_elo['mein Elo'])/len(df_elo['mein Elo']),0))
except Exception:
    avg_elo = np.nan


col1, col2, col3, col4 = st.columns(4)
col1.metric( label = 'aktueller Elo', value = letzter_elo, delta = elo_delta, help= 'Das Delta wird zu den letzten 15 Spielen gebildet' )
col2.metric( label= 'höchster Elo', value= max_elo, delta= max_delta, help= 'Das Delta wird zum aktuellen Elo gebildet' )
col3.metric( label= 'tiefster Elo', value= min_elo, delta= min_delta, help= 'Das Delta wird zum aktuellen Elo gebildet' )
col4.metric( label= 'Ø Elo', value= avg_elo, delta= "{:.0%}".format( (letzter_elo - avg_elo ) / avg_elo ), help= 'Das Delta wird zum aktuellen Elo gebildet')

fig = px.line(
    df_elo,
    y=["Gegner Elo", "mein Elo"], 
    title='Elo nach Spiel-Art'
    )
fig.update_layout(
    yaxis_title="Elo",
    xaxis_title="nach Spielen",
    legend_traceorder="reversed",
    legend_title_text='Elo'
)
fig.update_layout(hovermode='x unified')
fig.update_traces(hovertemplate="%{y} Elo<extra></extra>")
fig.for_each_trace(lambda trace: trace.update(visible="legendonly") if trace.name in "Gegner Elo" else ())
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})

opp_won_highest_name =  df_elo.sort_values(by='Gegner Elo', ascending=False)['Gegner-Name'][df_elo['Ausgang']=='gewonnen'].iloc[0]
opp_won_highest_elo =   df_elo.sort_values(by='Gegner Elo', ascending=False)['Gegner Elo'][df_elo['Ausgang']=='gewonnen'].iloc[0]
opp_won_highest_link =  df_elo.sort_values(by='Gegner Elo', ascending=False)['Link'][df_elo['Ausgang']=='gewonnen'].iloc[0]
opp_won_highest_datum = df_elo.sort_values(by='Gegner Elo', ascending=False)['Datum'][df_elo['Ausgang']=='gewonnen'].iloc[0]

opp_lost_lowest_name =  df_elo.sort_values(by='Gegner Elo')['Gegner-Name'][df_elo['Ausgang']=='verloren'].iloc[0]
opp_lost_lowest_elo =   df_elo.sort_values(by='Gegner Elo')['Gegner Elo'][df_elo['Ausgang']=='verloren'].iloc[0]
opp_lost_lowest_link =  df_elo.sort_values(by='Gegner Elo')['Link'][df_elo['Ausgang']=='verloren'].iloc[0]
opp_lost_lowest_datum =  df_elo.sort_values(by='Gegner Elo')['Datum'][df_elo['Ausgang']=='verloren'].iloc[0]

col1, col2 = st.columns(2)
col1.write('**👍 Höchster Elo gegen den du gewonnen hast:**')
col1.write(opp_won_highest_name + ' mit einem Elo von ' + str(opp_won_highest_elo) + ' am ' + str(pd.to_datetime(opp_won_highest_datum).strftime('%d.%m.%Y'))
    + ' [(zum Spiel)](' + opp_won_highest_link + ')')
col2.write('**👎 Tiefster Elo gegen den du verloren hast:**')
col2.write(opp_lost_lowest_name + ' mit einem Elo von ' + str(opp_lost_lowest_elo) + ' am ' + str(pd.to_datetime(opp_lost_lowest_datum).strftime('%d.%m.%Y'))
    + ' [(zum Spiel)](' + opp_lost_lowest_link + ')')










st.header('Spielvolumen im Zeitverlauf')

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    label='Jahre seit erstem Spiel',
    value= str(round((pd.to_datetime(max(df['Datum']))-pd.to_datetime(min(df['Datum']))).days/365.25,1)).replace('.',',')
)
col2.metric(
    label='Tage mit min. einem Spiel',
    value= "{:.0%}".format(
        int(len(
            pd.to_datetime(df['Datum']).apply(lambda x: x.date()).unique() ) ) / 
            (pd.to_datetime(max(df['Datum']))-pd.to_datetime(min(df['Datum']))).days
        )
)
col3.metric(
    label='Spiele/Tag',
    value= str( round( len(df) / (pd.to_datetime(max(df['Datum']))-pd.to_datetime(min(df['Datum']))).days , 2)).replace('.',',')
)
col4.metric(
    label='Spiele/Tag (min. ein Spiel)',
    value= str( round(len(df) / int( len( pd.to_datetime(df['Datum']).apply(lambda x: x.date()).unique() ) ), 2)).replace('.',',')
)

fig = px.histogram(
    df, 
    x ='Datum',
    title='im gesamten Zeitverlauf'
    )
fig.update_layout(yaxis_title="Anzahl Spiele")
fig.update_layout(hovermode='x unified')
fig.update_traces(hovertemplate="%{y} Spiele<extra></extra>")
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})


fig = px.histogram(
    df, 
    x ='Wochentag', 
    category_orders={"Wochentag":["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]},
    title='im Wochenverlauf'
    )
fig.update_layout(
    yaxis_title="Anzahl Spiele" 
)
fig.update_traces(hovertemplate="%{y} Spiele<extra></extra>")
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})


# Convert string column to datetime.time
def str_to_time(time_str):
  return datetime.strptime(time_str, '%H:%M:%S').time()

# Apply the function to the column
df['Uhrzeit'] = df['Uhrzeit'].apply(str_to_time).apply(lambda x: datetime.combine(datetime(1998, 11, 2), x))


fig = px.histogram(
    df, 
    x = df['Uhrzeit'],
    title='im Tagesverlauf'
    )
fig.update_layout(
    yaxis_title="Anzahl Spiele",
    xaxis_title="Uhrzeit"
)
fig.update_layout(hovermode='x unified')
fig.update_traces(hovertemplate="%{y} Spiele<extra></extra>")
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})

day_most_games =        df.groupby(['Wochentag']).size().sort_values(ascending=False).index[0]
day_least_games =       df.groupby(['Wochentag']).size().sort_values(ascending=True).index[0]
num_day_most_games =    df.groupby(['Wochentag']).size().sort_values(ascending=False)[0]
num_day_least_games =   df.groupby(['Wochentag']).size().sort_values()[0]
diff_day_games =        "{:.0%}".format((num_day_most_games-num_day_least_games)/num_day_least_games)
st.caption('am häufigsten spielst du am ' + day_most_games + ' mit ' + diff_day_games + ' mehr Spielen als am ' + day_least_games + ' mit den wenigsten Spielen')



st.subheader('Dynamische Visualisierungen')

cat_feat = st.selectbox(
    'Wähle kategorische Variable', 
    df.select_dtypes(exclude = 'number').columns)

angezeigteWerte3 = st.radio(
    "Werte",
    ('Absolut', 'Relativ'),
    horizontal=True
    )

if angezeigteWerte3 == 'Absolut':
    _var_angezeigteWerte3 = ''
else:
    _var_angezeigteWerte3 = 'percent'

fig = px.histogram(
    df, 
    x =cat_feat, 
    color = 'Ausgang',
    barnorm=_var_angezeigteWerte3,
    color_discrete_map={
        "gewonnen": "#007B57",
        "unentschieden": "grey",
        "verloren": "#9d2e00"}, 
    category_orders={
        "Ausgang":["gewonnen", "unentschieden", "verloren"], 
        "Wochentag":["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
        "Spiel-Art":["Bullet", "Blitz", "Schnellschach", "Täglich"],
        "Zeit":[
            "20 Sek. | 1", "30 Sek.", "1 Minute", "1 | 1", "2 | 1", 
            "3 Minuten", "3 | 2", "5 Minuten", "5 | 5",
            "10 Minuten", "10 | 5", "10 | 10", "15 Minuten", "15 | 5", "15 | 10", "15 | 15", "20 Minuten", "30 Minuten",
            "1 Tag", "2 Tage", "3 Tage", "7 Tage"
            ]
        },
    )
fig.update_layout(
    yaxis_title="Anzahl Spiele" 
)
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})



st.subheader('Ausgang meiner Spiele')

streaks = pd.Series(df.Ausgang, name='Ausgang').to_frame()

def generate_streak_info(shots):
    data = shots['Ausgang'].to_frame()
    data['start_of_streak'] = data['Ausgang'].ne(data['Ausgang'].shift())
    data['streak_id'] = data.start_of_streak.cumsum()
    data['Serie'] = data.groupby('streak_id').cumcount() + 1
    shots_with_streaks = pd.concat([shots, data['Serie']], axis=1)
    return shots_with_streaks

shots = streaks['Ausgang'].to_frame()
generate_streak_info(shots).groupby(by="Ausgang").max()
df_streak = generate_streak_info(shots).groupby(by="Ausgang").max()

col1, col2, col3 = st.columns(3)
col1.metric('längste Siegesserie', df_streak.loc['gewonnen'])
col2.metric('längste Unentschieden-Serie', df_streak.loc['unentschieden'])
col3.metric('längste Niederlagen-Serie', df_streak.loc['verloren'])







st.subheader('Einfluss der Farbe auf das Spiel')


col1, col2, col3 = st.columns(3)

col1.metric(
    'gewonnen',
    "{:.0%}".format(len(df[ (df['meine Farbe']=='Weiß') ].loc[(df[ (df['meine Farbe']=='Weiß') ].Ausgang == "gewonnen")]) / len(df[ (df['meine Farbe']=='Weiß') ].index))
    )
col2.metric(
    'unentschieden',
    "{:.0%}".format(len(df[ (df['meine Farbe']=='Weiß') ].loc[(df[ (df['meine Farbe']=='Weiß') ].Ausgang == "unentschieden")]) / len(df[ (df['meine Farbe']=='Weiß') ].index))
    )
col3.metric(
    'verloren',
    "{:.0%}".format(len(df[ (df['meine Farbe']=='Weiß') ].loc[(df[ (df['meine Farbe']=='Weiß') ].Ausgang == "verloren")]) / len(df[ (df['meine Farbe']=='Weiß') ].index))
    )





col1, col2, col3 = st.columns(3)

col1.metric(
    'gewonnen',
    "{:.0%}".format(len(df[ (df['meine Farbe']=='Schwarz') ].loc[(df[ (df['meine Farbe']=='Schwarz') ].Ausgang == "gewonnen")]) / len(df[ (df['meine Farbe']=='Schwarz') ].index))
    )
col2.metric(
    'unentschieden',
    "{:.0%}".format(len(df[ (df['meine Farbe']=='Schwarz') ].loc[(df[ (df['meine Farbe']=='Schwarz') ].Ausgang == "unentschieden")]) / len(df[ (df['meine Farbe']=='Schwarz') ].index))
    )
col3.metric(
    'verloren',
    "{:.0%}".format(len(df[ (df['meine Farbe']=='Schwarz') ].loc[(df[ (df['meine Farbe']=='Schwarz') ].Ausgang == "verloren")]) / len(df[ (df['meine Farbe']=='Schwarz') ].index))
    )






st.caption('Mit Liebe gebaut von [Özgün Cakir](https://www.özgüncakir.de), siehe auch das zugehörige [GitHub-Repo](https://github.com/OezguenCakir/Schach-Streamlit)')

num_games_measured = len(df[df['Meine Genauigkeit']!='nan%'])

st.subheader(str(num_games_measured) + ' Spiele hat Chess.com bzgl. eurer Genauigkeit gemessen')

st.caption('Dies misst die Übereinstimmung der gespielten Spielzüge mit Computer-berechneten Zügen')
st.write(df[df['Meine Genauigkeit']!='nan%'])



if st.button('Drück mich :)'):
    st.balloons()