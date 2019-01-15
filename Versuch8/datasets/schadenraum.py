from uncertainties import ufloat
from uncertainties.umath import *
from numpy import pi

import sys
sys.path.insert(0,"../../scripts")
from myPlot import ShorthandFormatter

'''=================================================================
Das sind die Masstabe vom Gasthermometer uns schaedlicher Raum.
Ich hab die Volumina nicht in Excel gerechnet, weil die Fehlerrechnung
mega muehsam wird und da kann ich es automatisch machen.
----------------------------------------------------------------
Die Werte in der Excel Tabelle sind nur als Referenz da,
aber falls etwas geaendert werden muss, muss man es hier machen.
====================================================================='''

#Schaedlicher Raum in mm
l1 = ufloat(44,0.5) #hoehe des ersten senkrechten Teils des Rohres
l2 = ufloat(177,1) #laenge des wagerechten Teils Rohres
l3 = ufloat(58.2,0.5) #hoehe des zweiten senkrechten Teils des Rohres, wo die Kugel ist
dt = ufloat(7,0.05) #externer Duchmesser des Rohres
st = ufloat(1.5,0.5) #Breite der Wand des Rohres
dc = ufloat(13,0.05) #externer Durchmesser des Quecksilber enthaltenden Rohres
sc = ufloat(1.5,1) #Breite der Wand des Quecksilber enthaltenden Rohres
h1 = ufloat(20.5,1) #Hoehe des Geraden Teils des Quecksilber enthaldenden Rohres
h2 = ufloat(13,1) #Hoehe des Kegelfoermigen Teils des Quecksilber enthaltenden Rohres
vs = pi*((dt/2-st)**2*((l1-dt/2)+(l2-2*dt)+(l3-dt/2))+h1*(dc/2-sc)**2+h2/3*(dc/2)**2)
#vs = pi*((dt/2-st)**2*((l1-st)+(l2-2*st)+(l3-st))+h1*(dc/2-sc)**2)

#Kugel in mm
dk = ufloat(66.7, 0.1) #externer Durchmesser der Kugel
sk = ufloat(0.5,0.5) #Breite der Wand der Kugel
vk = pi*4/3*(dk/2-sk)**3

#nur Formatierung
fmtr = ShorthandFormatter()
vk_str = fmtr.format('{0:.1u}', vk/1000) #von mm^3 auf ml
vs_str = fmtr.format('{0:.1u}', vs/1000)

print ("Das Volumen der Kugel ist " + vk_str + ' ml')
print ("Das Volumen der schaedlichen Raums ist " + vs_str + ' ml')
