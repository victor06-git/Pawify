from dimos import Dimos

if __name__ == "__main__":
    app = Dimos(n_workers=8)
    app.run("unitree-go2")
    
    app.skills.relative_move(forward=3.0)
    try:
        input("Blueprint running — press Enter to stop...\n")
    finally:
        app.stop()