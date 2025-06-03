import vtk
from vtk import vtkPLYReader, vtkPolyData, vtkPolyDataMapper, vtkRenderer
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderWindowInteractor, vtkActor
from vtkmodules.vtkCommonDataModel import vtkIterativeClosestPointTransform
from vtkmodules.vtkFiltersCore import vtkTransformPolyDataFilter

# load data
plyReader = vtkPLYReader()
plyReader.SetFileName("bunny.ply") #Putanja do željene datoteke
plyReader.Update() #Učitaj
sourcePD = plyReader.GetOutput() #Učitana geometrija se nalazi u vtkPolyData objektu
targetPD = vtkPolyData()

icp = vtkIterativeClosestPointTransform()
icp.SetSource(sourcePD) #Ulazni objekt (početna poza objekta)
icp.SetTarget(targetPD) #Konačni objekt (željena poza objekta)
icp.GetLandmarkTransform().SetModeToRigidBody() #Potrebni način rada je transformacija za kruta tijela
icp.SetMaximumNumberOfIterations(20) #Željeni broj iteracija
icp.SetMaximumNumberOfLandmarks(1000) #Koliko parova točaka da se koristi prilikom minimiziranja cost funkcije
icp.Update() #Provedi algoritam
icpTransformFilter = vtkTransformPolyDataFilter()
icpTransformFilter.SetInputData(source) #Objekt s početnim koordinatama
icpTransformFilter.SetTransform(icp) #transformiramo na novi položaj koristeći transformacijsku matricu
icpTransformFilter.Update()
icpResultPD = icpTransformFilter.GetOutput() #Transformirani (novi) objekt


# rendering
renderer = vtkRenderer()
renderer.SetBackground(1.0, 1.0, 1.0) #Bijela pozadina
renderer.AddActor(sphereActor) #Dodaj neki objekt na scenu
renderer.ResetCamera() #Centriraj kameru tako da obuhvaća objekte na sceni

# render window
window = vtkRenderWindow()
window.AddRenderer(renderer) #Moguće je dodati i više renderera na jedan prozor
window.SetSize(800, 600) #Veličina prozora na ekranu
window.SetWindowName("Scena") #Naziv prozora
window.Render() #Renderaj scenu

# interactor
interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)

interactor.Start() #Pokretanje interaktora, potrebno kako se vtk prozor ne bi odmah zatvorio

# mapper
mapper = vtkPolyDataMapper()
mapper.SetInputData(pd) #Ulazni podatak je objekt tipa vtkPolyData

# actor (object)
actor = vtkActor()
actor.SetMapper(mapper) #Povezujemo ga s mapperom za određeni tip podataka
actor.GetProperty().SetColor(1.0, 0.0, 0.0) #Objekt će biti obojan u crveno
actor.GetProperty().SetPointSize(5) #Veličina će biti 5x5 piksela po točci
renderer.AddActor(actor) #Dodajemo ga na popis objekata na sceni