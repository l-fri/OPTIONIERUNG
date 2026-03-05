
## Überblick

Dieses Projekt besteht aus vier Python-Skripten. Jedes Skript liest einen CSV-Export (Transaction History) ein und erzeugt (1) eine Konsolen-Ausgabe mit Kennzahlen in EUR und (2) ein Diagramm als PNG-Datei.

Allgemeines Ausführungs-Muster:

```bash
python <script>.py --csv <pfad_zur_csv>

Beispiel (CSV muss in data liegen)

python options_data.py --csv data/U24066232.TRANSACTIONS.YTD.csv

python stocks_data.py --csv data/U24066232.TRANSACTIONS.YTD.csv

python cash_fx_data.py --csv data/U24066232.TRANSACTIONS.YTD.csv

python interest.py --csv data/U24066232.TRANSACTIONS.YTD.csv