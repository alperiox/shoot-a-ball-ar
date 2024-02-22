
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# get the current camera

position = (600, 400)
radius = 15

circle_check = lambda x, y: (x - position[0]) ** 2 + (y - position[1]) ** 2 <= radius**2

interaction_radius = 45
interaction_check = (
    lambda x, y: (x - position[0]) ** 2 + (y - position[1]) ** 2
    < interaction_radius**2 - 5
)
acceleration = 0
velocity = 0
friction = 0.5
g_vector = (0, -20)

# check flags
draw_line_check = False
ball_shot_check = False

# variables to store the previous locations of thumb and index finger
prev_thumb_coords = None
prev_index_coords = None

# ellipse constants (aka the hoop)
hoop_position = (1000, 250)
hoop_radius = (60, 25)
hoop_angle = -30
hoop_color = (0, 0, 255)
hoop_thickness = 2

# check for the hoop
hoop_equation = (lambda x, y: 
    (x - hoop_position[0]) ** 2 / hoop_radius[0] ** 2
    + (y - hoop_position[1]) ** 2 / hoop_radius[1] ** 2)
hoop_check = lambda x, y: ( hoop_equation(x,y) <= 1 )
hoop_flag = False
score = 0

debug = False