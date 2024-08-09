# I can improve it

# Define variables
BROWSER = open  # firefox, safari, etc.
URL = https://chatgpt.com/ #https://www.google.com

# Default target
all: open_website run_main

# Target to open a website
open_website:
	@echo "Opening website $(URL)..." 
	$(BROWSER) $(URL)

# Target to run Python code
run_main:
	@echo "Running Python code..."
	python main.py

# 

build:
	@echo "Building the project..."
	python setup.py 

# Clean target (optional)
clean:
	@echo "Cleaning up..."
	rm -rf *.pyc