from gui import *

if __name__ == '__main__':
    app = QApplication([])
    cube = create_cube(8)
    cube_axis = GLAxisItem(size=[8, 8, 8], width=5)
    axis = GLAxisItem(size=[10, 10, 10], width=5)
    window = Gui3d(static_obj=[axis], dynamic_obj=[cube, cube_axis])
    window.show()
    app.exec_()
