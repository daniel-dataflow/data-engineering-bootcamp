import folium

my_loc = folium.Map(location=[37.5481533, 127.0733985], zoom_start=20)


folium.Marker(location=[37.5481533, 127.0733985],popup=folium.Popup("멀티캠퍼스 세종", max_width=100)).add_to(my_loc)

my_loc.save("visual02.html")