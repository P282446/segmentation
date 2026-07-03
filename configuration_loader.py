import configparser
import ast

config = configparser.ConfigParser()
config.read('configurations/config.txt')
eyes_region = ast.literal_eval(config["regions"]["eyes_region"])

mouth_region = ast.literal_eval(config["regions"]["mouth_region"])

face_region = ast.literal_eval(config["regions"]["face_region"])
