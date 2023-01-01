import os
import pandas as pd
import chess.pgn
import urllib
import urllib.request
import json
import re
import os.path
import pathlib
import logging
from datetime import datetime
import time
import traceback
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

os.chdir('/Users/ozguncakir/Schach_Streamlit/Schach_Streamlit')


def callback():
    # Button wurde geklickt
    st.session_state.button_clicked = True

def delete_pgn_files():
    for f in os.listdir('/Users/ozguncakir/Schach_Streamlit/Schach_Streamlit/data'):
        os.remove(os.path.join('/Users/ozguncakir/Schach_Streamlit/Schach_Streamlit/data', f))

@st.cache(suppress_st_warning=True, show_spinner=False)
def download_pgn_files():
    baseUrl = "https://api.chess.com/pub/player/" + username + "/games/"
    progressbar = st.progress(0)
    # read the archives url and store in a list
    f = urllib.request.urlopen(baseUrl + "archives")
    archivesList = f.read().decode("utf-8").replace("{\"archives\":[\"", "\",\"").split("\",\"" + baseUrl)
    archivesList[len(archivesList)-1] = archivesList[len(archivesList)-1].rstrip("\"]}")
    # download all the archives
    p = st.empty()
    for i in range(len(archivesList)-1):
        url = baseUrl + archivesList[i+1] + "/pgn"
        filename = archivesList[i+1].replace("/", "-")
        urllib.request.urlretrieve(url, "data/" + filename + ".pgn")
        p.caption(filename + ".pgn wird heruntergeladen")
        i = i+ 1
        progressbar.progress( (i+1)/len(archivesList) )
         
@st.cache(suppress_st_warning=True, show_spinner=False)
def create_join_json_file():
    log = logging.getLogger().error
    inp_dir = pathlib.Path('data/')
    out_dir = pathlib.Path('')

    def get_file_list(local_path):
        tree = os.walk(str(local_path))
        file_list = []
        out = []
        test = r'.+pgn$'
        for i in tree:
            file_list = i[2]

        for name in file_list:
            if(len(re.findall(test, name))):
                out.append(str(local_path / name))
        return out

    def get_data(pgn_file):
        node = chess.pgn.read_game(pgn_file)
        while node is not None:
            data = node.headers
            data["moves"] = []
            while node.variations:
                next_node = node.variation(0)
                data["moves"].append(
                        re.sub("\{.*?\}", "", node.board().san(next_node.move)))
                node = next_node
            node = chess.pgn.read_game(pgn_file)
            out_dict = {}

            for key in data.keys():
                out_dict[key] = data.get(key)
            yield out_dict

    def convert_file(file_path):
        file_name = file_path.name.replace(file_path.suffix, '') + '.json'
        st.write('convert file '+file_path.name)
        out_list = []
        try:
            json_file = open(str(out_dir / file_name), 'w')
            pgn_file = open(str(file_path), encoding='ISO-8859-1')

            for count_d, data in enumerate(get_data(pgn_file), start=0):
                st.write(file_path.name+' '+str(count_d))
                out_list.append(data)

            st.write(' save '+file_path.name)
            json.dump(out_list, json_file)
            json_file.close()
            st.write('done')
        except Exception as e:
            st.write(traceback.format_exc(10))
            st.write('ERROR file '+file_name+' not converted')

    def create_join_file(file_list):
        name = str(out_dir / 'chess_data.json')
        open(name, 'w').close()
        json_file = open(str(out_dir / 'chess_data.json'), 'a')
        json_file.write('[')
        progressbar = st.progress(0)
        p = st.empty()
        for count_f, file in enumerate(file_list, start=0):
            pgn_file = open(file, encoding='ISO-8859-1')
            for count_d, data in enumerate(get_data(pgn_file), start=0):
                log(str(count_f)+' '+str(count_d))
                if count_f or count_d:
                    json_file.write(',')
                data_str = json.dumps(data)
                json_file.write(data_str)
            progressbar.progress((count_f+1)/len(file_list))
            p.caption(pathlib.Path(file).name + " wird bearbeitet")
        json_file.write(']')
        json_file.close()

    file_list = get_file_list(inp_dir)
    file_list.sort()
    create_join_file(file_list)

def datenbearbeitung(df):
    import datetime
    df['meine Farbe'] = np.where(df['Black']!= text_input, 'Weiß', 'Schwarz')
    df['mein Elo'] = np.where(df['meine Farbe']== 'Weiß', df['WhiteElo'], df['BlackElo'])
    df['Gegner Elo'] = np.where(df['meine Farbe']== 'Weiß', df['BlackElo'], df['WhiteElo'])
    df['Gegner-Name'] = np.where(df['meine Farbe']== 'Weiß', df['Black'], df['White'])
    df['End-Datum'] = pd.to_datetime(df['EndDate'] + ' ' + df['EndTime'])
    df['Start-Datum'] = pd.to_datetime(df['UTCDate'] + ' ' + df['StartTime'])
    df['Dauer'] = df['End-Datum'] - df['Start-Datum']
    df['Jahr'] = pd.DatetimeIndex(df['Date']).year
    df['Dauer in Minuten'] = round(df['Dauer'] / datetime.timedelta(minutes=1),2)
    if df.get('variant') == None:
        df['Variante'] = 'Classic'
    else: df['Variante'] = df.get('Variant', 'leer').replace("leer","Classic").fillna("Classic")
    df['Anzahl Züge'] = np.ceil(df.moves.apply(len)/2).astype('int')
    df['Uhrzeit'] = pd.to_datetime(df.StartTime, format="%H:%M:%S")

    conditions = [
        (df['Result'] == '1/2-1/2'),
        (df['Result'] == '0-1'),
        (df['Result'] == '1-0')
        ]
    values = [0.5, 0, 1]
    values_black = [0.5, 1, 0]

    df['Ausgang_kurz'] = np.where(df['meine Farbe']== 'Weiß', np.select(conditions, values), np.select(conditions, values_black))
    

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
    game_type_mapping = {
        "600":"Schnellschach",
        "60":"Bullet",
        "60+1":"Bullet",
        "300":"Blitz",
        "1/172800":"Täglich",
        "180":"Blitz",
        "1800":"Schnellschach",
        "120+1":"Bullet",
        "900+5":"Schnellschach",
        "900":"Schnellschach",
        "300+5":"Blitz",
        "180+2":"Blitz",
        "30":"Bullet",
        "900+10":"Schnellschach",
        "1/86400":"Täglich"
    }
    df['Spiel-Art'] = df['TimeControl'].astype(str).map(game_type_mapping)

    game_result_mapping = {
        1:"gewonnen",
        0:"verloren",
        0.5:"unentschieden"
    }
    df['Ausgang'] = df['Ausgang_kurz'].map(game_result_mapping)

    conditions = [
        df.Termination.str.contains("won by resignation"),
        df.Termination.str.contains("won by checkmate"),
        df.Termination.str.contains("won - game abandoned"),
        df.Termination.str.contains("won on time"),
        df.Termination.str.contains("drawn by agreement"),
        df.Termination.str.contains("drawn by repetition"),
        df.Termination.str.contains("drawn by insufficient material"),
        df.Termination.str.contains("drawn by stalemate"),
        df.Termination.str.contains("drawn by timeout vs insufficient material"),
        df.Termination.str.contains("drawn by 50-move rule"),
        df.Termination.str.contains("won by checking the opponent king for the 3rd time")
        ]

    values = ['Aufgabe', 'Schachmatt', 'Verlassen', 'Zeitüberschreitung', 'Vereinbarung', 'Stellungswiederholung', 'Nicht genügend Material', 'Patt', 'Zeitüberschreitung gegen unzureichendes Material', '50 Züge Regel', '3 Mal Schach-gestellt']
    df['Ausgang-Grund'] = np.select(conditions, values)

    df['Event'] = df.Event.astype('category')
    df['Spiel-Art'] = df['Spiel-Art'].astype('category')
    df['Ausgang'] = df.Ausgang.astype('category')
    df['Ausgang-Grund'] = df['Ausgang-Grund'].astype('category')


    from datetime import datetime
    df['Wochentag Zahl'] = df['Start-Datum'].apply(lambda time: time.dayofweek)
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
    df['Wochentag']= df['Wochentag Zahl'].map(wochentag_mapping)
    df['Start-Datum'] = pd.to_datetime(df['Start-Datum'])
    df['End-Datum'] = pd.to_datetime(df['End-Datum'])

    df = df[[
        'Event', 'Spiel-Art','Variante', 'Zeit', 
        'Start-Datum', 'End-Datum', 'Dauer in Minuten',
        'Ausgang', 'Ausgang-Grund', 'meine Farbe', 'mein Elo', 'Gegner Elo', 'Gegner-Name', 'Anzahl Züge', 'ECO', 'Uhrzeit', 'Wochentag']]
    df = df.sort_values('Start-Datum', ascending=False).reset_index(drop=True)
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
    erstes_datum = pd.to_datetime( min(df['Start-Datum']) )
    return erstes_datum

def function_letztes():
    letztes_datum = pd.to_datetime( max(df['Start-Datum']) )
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



st.title('♟️ Chess.com Profil-Analyse')


with st.form(key='my_form'):
	text_input = st.text_input(label='**Gebe deinen Chess.com Usernamen ein**')
	submit_button = st.form_submit_button(label='anfangen', on_click=callback)
if submit_button or st.session_state.button_clicked:
    username = text_input.lower()

    # Datendownload
    st.markdown('**Monatliche Datenstände aus Chess.com werden als PGN-Dateien heruntergeladen**')  
    delete_pgn_files()
    download_pgn_files()           
    st.success ("Alle Dateien wurden heruntergeladen ✅")
    
    # Datenzusammenführung
    st.markdown("**Jetzt werden alle Spiele aus den Dateien zusammengeführt**")  
    create_join_json_file()
    delete_pgn_files()
    st.success('Die Zusammenführung der Dateien ist fertig ✅')

    # Datenbearbeitung und direkt als df speichern
    df = pd.read_json('chess_data.json')
    df = datenbearbeitung(df)
    st.text('')
    st.info('So... und jetzt geht es los  \n Falls du einen neuen Namen eingeben willst, musst du den Cache refreshen (drücke `C`)')
else:
    st.subheader('Oder schaue dir meine Daten an')
    username = 'oezguen'
    df = pd.read_json('my_chess_data.json')
    df = datenbearbeitung(df)

with st.sidebar:
    st.subheader('Filter')

    radio_spielart = st.radio(
        "Spiel-Art",
        ("nur klassisches Schach", "alle Varianten"),
        help='es gibt besondere Schach.Varianten mit unterschiedlichen Zielen. Dieser Filter blendet diese aus'
    )
    if radio_spielart == 'nur klassisches Schach':
        df = df[ (df.Variante=='Classic') & (df.Event=='Live Chess') ]
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
            min_value = pd.to_datetime( min(df['Start-Datum']) ),
            max_value = pd.to_datetime( max(df['Start-Datum']) ),
            value = (pd.to_datetime(min(df['Start-Datum'])), pd.to_datetime(max(df['Start-Datum'])) )
        )
    df = df[ ( df['Start-Datum'] >= function_erstes() ) & (df['Start-Datum'] <= function_letztes() ) ]
    

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


anzahl_spiele = len(df.index)
anzahl_spiele_gewonnen = len( df.loc[(df.Ausgang == "gewonnen")] )
anzahl_spiele_verloren = len( df.loc[(df.Ausgang == "verloren")] )
anzahl_spiele_unentschieden = len( df.loc[(df.Ausgang == "unentschieden")] )

quote_spiele_gewonnen = "{:.0%}".format(anzahl_spiele_gewonnen / anzahl_spiele)
quote_spiele_verloren = "{:.0%}".format(anzahl_spiele_verloren / anzahl_spiele)
quote_spiele_unentschieden = "{:.0%}".format(anzahl_spiele_unentschieden / anzahl_spiele)


url = "https://api.chess.com/pub/player/" + username
data = json.loads(urllib.request.urlopen(url).read())
col1, col2 = st.columns([1,5])
if data.get('avatar') == None:
    profile_pic = 'https://www.chess.com/bundles/web/images/user-image.007dad08.svg'
else:
    profile_pic = data.get('avatar')

col1.image(profile_pic, width=100)
col2.write("[" + data.get('username') + "](" + data.get('url') + ")  \n" + data.get('name'))
col2.caption("Follower: " + str(data.get('followers')))
col2.caption(
    "Zuletzt online am " + time.strftime('%d.%m.%Y um %H:%M:%S', time.localtime(data.get('last_online'))) + "  \n" + 
    "Registriert am " + time.strftime('%d.%m.%Y um %H:%M:%S', time.localtime(data.get('joined')))
    )

st.subheader("Der Datensatz")


st.dataframe(filter_dataframe(df))


st.subheader('Statistik numerischer Daten')

st.dataframe(df.describe().applymap(lambda x: f"{x:0.2f}"))


st.subheader('Anzahl Spiele über die Jahre')

col1, col2, col3, col4 = st.columns(4)
col1.metric('Anzahl Spiele insgesamt', anzahl_spiele)
col2.metric('gewonnen', quote_spiele_gewonnen)
col3.metric('unentschieden', quote_spiele_unentschieden)
col4.metric('verloren', quote_spiele_verloren)

angezeigteWerte = st.radio(
    "Angezeigte Werte",
    ('Absolut', 'Relativ'),
    horizontal=True
    )

if angezeigteWerte == 'Absolut':
    _var_angezeigteWerte = ''
else:
    _var_angezeigteWerte = 'percent'


fig = px.histogram(
    df, 
    x ='Start-Datum', 
    color = 'Ausgang', 
    barnorm=_var_angezeigteWerte,
    color_discrete_map={
        "gewonnen": "#007B57",
        "unentschieden": "grey",
        "verloren": "#9d2e00"}, 
    category_orders={"Ausgang":["gewonnen", "unentschieden", "verloren"]}
    )
fig.update_layout(
    yaxis_title="Anzahl Spiele"
)
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})



st.subheader('Wann ich spiele')


fig = px.histogram(
    df, 
    x ='Uhrzeit',
    title='im Tagesverlauf'
    )
fig.update_layout(
    yaxis_title="Anzahl Spiele" 
)
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
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})




st.subheader('Elo nach Spielen')



radio_spielart = st.radio(
        "Spiel-Art",
        ("Bullet", "Blitz", "Schnellschach", "Täglich"),
        horizontal=True,
        help='Dieser Filter wirkt nur auf diese Visualisierung ein',
        index=2
    )
if radio_spielart == 'Bullet':
    df_elo = df[ (df['Spiel-Art']=='Bullet') ].sort_values('Start-Datum').reset_index(drop=True)
elif radio_spielart == 'Blitz':
    df_elo = df[ (df['Spiel-Art']=='Blitz') ].sort_values('Start-Datum').reset_index(drop=True)
elif radio_spielart == 'Schnellschach':
    df_elo = df[ (df['Spiel-Art']=='Schnellschach') ].sort_values('Start-Datum').reset_index(drop=True)
elif radio_spielart == 'Täglich':
    df_elo = df[ (df['Spiel-Art']=='Täglich') ].sort_values('Start-Datum').reset_index(drop=True)



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

col1.metric(
    label = 'aktueller Elo', 
    value = letzter_elo,
    delta = elo_delta,
    help= 'Das Delta wird zu den letzten 15 Spielen gebildet'
    )

col2.metric(
    label= 'höchster Elo',
    value= max_elo,
    delta= max_delta,
    help= 'Das Delta wird zum aktuellen Elo gebildet'
)

col3.metric(
    label= 'tiefster Elo',
    value= min_elo,
    delta= min_delta,
    help= 'Das Delta wird zum aktuellen Elo gebildet'
)

col4.metric(
    label= 'Ø Elo',
    value= avg_elo,
    delta= "{:.0%}".format( (letzter_elo - avg_elo ) / avg_elo ),
    help= 'Das Delta wird zum aktuellen Elo gebildet'
)



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
fig.for_each_trace(lambda trace: trace.update(visible="legendonly") 
                   if trace.name in "Gegner Elo" else ())
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})


st.subheader('Dynamische Visualisierungen')


num_feat = st.selectbox(
    'Wähle numerische Variable', 
    df.select_dtypes('number').columns)

angezeigteWerte2 = st.radio(
    "Werte  ",
    ('Absolut', 'Relativ'),
    horizontal=True
    )

if angezeigteWerte2 == 'Absolut':
    _var_angezeigteWerte2 = ''
else:
    _var_angezeigteWerte2 = 'percent'

fig = px.histogram(
    df, 
    x = num_feat, 
    color = 'Ausgang',
    barnorm=_var_angezeigteWerte2,
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
            "10 Minuten", "10 | 5", "10 | 10", "15 Minuten", "15 | 10", "30 Minuten",
            "1 Tag", "2 Tage", "3 Tage", "7 Tage"
            ]
        },
    )
fig.update_layout(
    yaxis_title="Anzahl Spiele" 
)
st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})



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
            "10 Minuten", "10 | 5", "10 | 10", "15 Minuten", "15 | 10", "30 Minuten",
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


fig = px.bar(
    df,
    y='Ausgang-Grund', 
    title='Wie ich Spiele gewinne und verliere',
    barmode="group",
    facet_col='Ausgang'
    )  
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.update_xaxes(visible=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# fig.update_yaxes(visible=False)

st.plotly_chart(fig, use_container_width=True, config= {'displaylogo': False})