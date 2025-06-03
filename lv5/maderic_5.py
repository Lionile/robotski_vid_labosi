import vtk
import time
from vtkmodules.vtkIOPLY import vtkPLYReader
from vtkmodules.vtkCommonDataModel import vtkIterativeClosestPointTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkRenderingCore import (
    vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor
)

def load_ply(filename):
    reader = vtkPLYReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# load bunnies
base_bunny = load_ply('3D_modeli/bunny.ply')
bunny_files = ['3D_modeli/bunny_t1.ply', '3D_modeli/bunny_t2.ply', '3D_modeli/bunny_t3.ply',
               '3D_modeli/bunny_t4_parc.ply', '3D_modeli/bunny_t5_parc.ply',]

windows = []
interactors = []

for i, fname in enumerate(bunny_files):
    target_bunny = load_ply(fname)
    icp = vtkIterativeClosestPointTransform()
    icp.SetSource(target_bunny)
    icp.SetTarget(base_bunny)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.SetMaximumNumberOfLandmarks(300)

    start_time = time.time()
    icp.Update()
    elapsed = time.time() - start_time
    print(f"ICP matching time for {fname}: {elapsed:.4f} seconds")

    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetInputData(target_bunny)
    transform_filter.SetTransform(icp)
    transform_filter.Update()

    # rendering
    renderer = vtkRenderer()
    renderer.SetBackground(1, 1, 1)

    # base bunny
    base_mapper = vtkPolyDataMapper()
    base_mapper.SetInputData(base_bunny)
    base_actor = vtkActor()
    base_actor.SetMapper(base_mapper)
    base_actor.GetProperty().SetColor(1,0,0)
    base_actor.GetProperty().SetPointSize(5)
    renderer.AddActor(base_actor)

    # matched bunny
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(transform_filter.GetOutput())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0,0,1)
    actor.GetProperty().SetPointSize(5)
    renderer.AddActor(actor)

    # render window
    window = vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(800, 600)
    window.SetWindowName(f"Bunny ICP Alignment {i+1}")

    # interactor
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    renderer.ResetCamera()
    window.Render()

    windows.append(window)
    interactors.append(interactor)

# start all
for interactor in interactors:
    interactor.Start()
