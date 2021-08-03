from qtpylib.blotter import Blotter

class MainBlotter(Blotter):
    pass # we just need the name

if __name__ == "__main__":
    blotter = MainBlotter()
    blotter.run()