import folium
import json

my_loc = folium.Map(location=[37.5481533, 127.0733985], zoom_start=13)

with open("starbucks.json", "r", encoding="utf-8") as file:
    starbucks_joson = json.load(file)

# print(starbucks_joson)
# print(starbucks_joson["list"])

for starbucks in starbucks_joson["list"]:
    folium.Marker(location=[starbucks["lat"], starbucks["lot"]],popup=folium.Popup(starbucks["s_name"],max_width=100)).add_to(my_loc)

my_loc.save("visual03.html")
