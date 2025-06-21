from app import create_app

app = create_app()

if __name__ == "__main__":
    # Puedes cambiar host/puerto si lo deseas:
    app.run(debug=True)