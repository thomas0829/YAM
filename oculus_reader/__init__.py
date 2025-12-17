# Make oculus_reader.oculus_reader accessible
import sys
import os

# Add the current directory to make nested oculus_reader importable
__path__.append(os.path.dirname(__file__))

