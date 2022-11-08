import pyglet.app

from envs.biarm import BiArmSim


def main():
    sim = BiArmSim()

    pyglet.app.run()
    # while True:
    #     sim.render()


if __name__ == "__main__":
    main()
