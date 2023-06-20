import numpy as np
import matplotlib.pyplot as plot

#Einlesen der Daten
#Reading in the data
h = 10**(-5)
t0 = 0
T = 100
t = h*np.arange(T/h+1)
x0 = np.transpose([0, -1, 0, 1])

B = np.identity(4)
A = np.array( ((0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0)) )
q = np.empty([4,int(T/h+1)])
q[0] = np.sin(t)

#Berechnen der Inversen Matrix 1/h*A+B
# Calculating the inverse Matrix 1/h*A+B
Inv = np.linalg.solve(1.0/h*A+B,np.identity(4))

#Erstellen eines leeren Arrays zur Speicherung der Approximation der Lösung der DAE
# Create an empty array to store the approximation of the solution of the DAE
x = np.empty([4,int(T/h+1)])
np.transpose(x)[0] = x0

#Berechnen der Approximation der Lösung mittels impliziten Euler
# Calculate the approximation of the solution using implicit Euler
for i in range(1, int(T/h+1)):
    np.transpose(x)[i] = np.dot(Inv,np.transpose(q)[i]+1.0/h*np.dot(A,np.transpose(x)[i-1]))

#Berechnen der exakten Lösung der DAE an den Stützstellen
#Calculate the exact solution of the DAE at the Support points
xsol = np.empty([4,int(T/h+1)])
xsol[0] = np.sin(t)
xsol[1] = -np.cos(t)
xsol[2] = -np.sin(t)
xsol[3] = np.cos(t)

e = x-xsol

#Plotten der komponentenweisen Fehler
#Plotting the component-by-component defects
plot.plot(t[np.arange(int(T/(10*h)))*10],e[0][np.arange(int(T/(10*h)))*10])
plot.xlabel('t')
plot.ylabel('e_1,n')
plot.title('global error for x_1')
plot.show()

plot.plot(t[np.arange(int(T/(10*h)))*10],e[1][np.arange(int(T/(10*h)))*10])
plot.xlabel('t')
plot.ylabel('e_2,n')
plot.title('global error for x_2')
plot.show()

plot.plot(t[np.arange(int(T/(100*h)))*100],e[2][np.arange(int(T/(100*h)))*100])
plot.xlabel('t')
plot.ylabel('e_3,n')
plot.title('global error for x_3')
plot.show()

plot.plot(t[np.arange(int(T/(1000*h)))*1000],e[3][np.arange(int(T/(1000*h)))*1000])
plot.xlabel('t')
plot.ylabel('e_4,n')
plot.title('global error for x_4')
plot.show()

