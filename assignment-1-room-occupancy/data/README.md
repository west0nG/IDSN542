# Room Occupancy Estimation

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)

Citation: Singh, A. & Chaudhari, S. (2018). *Room Occupancy Estimation* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P605.

License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

`Occupancy_Estimation.csv` is the unmodified file downloaded from UCI on September 15, 2026. The conversion from counts to binary occupancy happens in `occupancy.py`; the source file is unchanged.

There are 10,129 rows, 18 input columns (including date and time), and a room occupancy count. Each row is one sensor observation, not an independent person or room. The experiment uses six numeric sensor columns and uses date/time only for splitting and plotting.

| Column used | Meaning | Unit |
| --- | --- | --- |
| S1_Temp | Temperature at sensor 1 | °C |
| S5_CO2 | CO₂ concentration | ppm |
| S1_Light | Light at sensor 1 | lux |
| S1_Sound | Sound sensor amplifier output | volts, not decibels |
| S6_PIR, S7_PIR | Passive infrared motion detections | 0/1 |
| Room_Occupancy_Count | Recorded number of people | 0–3 |

The UCI narrative describes four days of controlled collection. The supplied CSV contains seven distinct dates: December 22–26, 2017 and January 10–11, 2018. This project follows the timestamps in the file. The observations come from one experimental room, with no HVAC operating during collection, according to UCI. Results do not establish performance in other rooms.
