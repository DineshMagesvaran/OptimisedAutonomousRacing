import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import math

class Graphics():
	def __init__(self):
		# CONSTANTS #
		self.track_thickness = 10
		self.hf_thickness = self.track_thickness/2 #half track thickness
		self.car_length = 2
		self.car_width = 1
		self.car_height = 1
		self.camera_follow_dist = 0.01
		self.camera_height = 50 #1.5
		self.camera_vertical_fov = 45 # field of view
		self.camera_max_view_dist = 100
		# VARIABLES #
		self.car_x = 0
		self.car_y = 0
		self.car_a = 0
		self.path = []
		self.track_layout = []

		# variables for manual WASD control
		self.car_speed = 0.2 # metre per frame
		self.car_handling = 1 # degrees per frame
		self.w_pressed_down = False
		self.a_pressed_down = False
		self.s_pressed_down = False
		self.d_pressed_down = False

		pygame.init()
		display = (400,400)
		self.window = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
		pygame.display.set_caption("Race Car Environment")
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(self.camera_vertical_fov, (display[0]/display[1]), 0.1, self.camera_max_view_dist)
		
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		self.setCamera(self.car_x, self.car_y, self.car_a)
	
	# RaceCarEnv calls this function to update the graphics every time step
	def updateGraphics(self, car_x, car_y, car_a, episode_no, speed, time_elapsed):
		self.car_x = car_x
		self.car_y = car_y
		self.car_a = car_a
		
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		self.setCamera(car_x, car_y, car_a)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		
		# Constructing a track via vertices
		track_layout = []
		track_layout.append([self.hf_thickness, 0])
		track_layout.append([-self.hf_thickness, 0])
		
		for i in range (101):
			x  = -20 + (20 + self.hf_thickness) * math.cos((i / 100) * math.pi)
			y  = 50 + (20 + self.hf_thickness) * math.sin((i / 100) * math.pi)
			track_layout.append([x, y])
			x  = -20 + (20 - self.hf_thickness) * math.cos((i / 100) * math.pi)
			y  = 50 + (20 - self.hf_thickness) * math.sin((i / 100) * math.pi)
			track_layout.append([x, y])
		
		for i in range (101):
			x  = -20 + (20 + self.hf_thickness) * math.cos((i / 100) * math.pi + math.pi)
			y  = 0 + (20 + self.hf_thickness) * math.sin((i / 100) * math.pi + math.pi)
			track_layout.append([x, y])
			x  = -20 + (20 - self.hf_thickness) * math.cos((i / 100) * math.pi + math.pi)
			y  = 0 + (20 - self.hf_thickness) * math.sin((i / 100) * math.pi + math.pi)
			track_layout.append([x, y])

		# Drawing floor, track and car
		self.draw_floor()
		self.draw_race_track(track_layout)
		self.draw_race_car(car_x, car_y, car_a)
	
		# Get image from the graphics
		size = self.window.get_size()
		buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)

		# Stats and path are drawn after the image is captured to prevent this from being part
		# of the image sent to the neural network
		self.draw_details(episode_no, speed, time_elapsed)
		self.draw_path()

		# Updating the graphics on screen
		pygame.display.flip()

		# Process and return image
		screen_surf = pygame.image.fromstring(buffer, size, "RGBA")
		imgdata = pygame.surfarray.array3d(screen_surf)
		dim = (96, 96)
		resized_imagdata = cv2.resize(imgdata, dim, interpolation = cv2.INTER_AREA)
		return resized_imagdata

	# Resets the graphics to the initial state
	def reset_graphics(self):
		self.path.clear()
		self.car_x = 0
		self.car_y = 0
		self.car_a = 0
		state_image = self.updateGraphics(self.car_x, self.car_y, self.car_a, 0, 0, 0)
		return state_image
	
	# This function is used for testing purposes only - It allows manual control of the car via W, A, S, D.
	def updateGraphicsManualInput(self):
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		self.setCamera(self.car_x, self.car_y, self.car_a)
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_w:
					self.w_pressed_down = True
				elif event.key == pygame.K_a:
					self.a_pressed_down = True
				elif event.key == pygame.K_s:
					self.s_pressed_down = True
				elif event.key == pygame.K_d:
					self.d_pressed_down = True
			elif event.type == pygame.KEYUP:
				if event.key == pygame.K_w:
					self.w_pressed_down = False
				elif event.key == pygame.K_a:
					self.a_pressed_down = False
				elif event.key == pygame.K_s:
					self.s_pressed_down = False
				elif event.key == pygame.K_d:
					self.d_pressed_down = False
		
		# Move car based on WASD controls
		if (self.w_pressed_down):
			self.car_x += self.car_speed * -math.sin(self.car_a * math.pi / 180)
			self.car_y += self.car_speed * math.cos(self.car_a * math.pi / 180)
		if (self.s_pressed_down):
			self.car_x += self.car_speed * math.sin(self.car_a * math.pi / 180)
			self.car_y += self.car_speed * -math.cos(self.car_a * math.pi / 180)
		if (self.a_pressed_down):
			self.car_a += self.car_handling
		if (self.d_pressed_down):
			self.car_a += -self.car_handling

		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		track_layout = []
		track_layout.append([2, 0])
		track_layout.append([-2, 0])
		
		for i in range (101):
			x  = -20 + 22 * math.cos((i / 100) * math.pi)
			y  = 50 + 22 * math.sin((i / 100) * math.pi)
			track_layout.append([x, y])
			x  = -20 + 18 * math.cos((i / 100) * math.pi)
			y  = 50 + 18 * math.sin((i / 100) * math.pi)
			track_layout.append([x, y])
		
		for i in range (101):
			x  = -20 + 22 * math.cos((i / 100) * math.pi + math.pi)
			y  = 0 + 22 * math.sin((i / 100) * math.pi + math.pi)
			track_layout.append([x, y])
			x  = -20 + 18 * math.cos((i / 100) * math.pi + math.pi)
			y  = 0 + 18 * math.sin((i / 100) * math.pi + math.pi)
			track_layout.append([x, y])

		self.draw_floor()
		self.draw_race_track(track_layout)
		self.draw_race_car(self.car_x, self.car_y, self.car_a)
		
		size = self.window.get_size()
		buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
		pygame.display.flip()
		pygame.time.wait(10)
		
		screen_surf = pygame.image.fromstring(buffer, size, "RGBA")
		imgdata = pygame.surfarray.array3d(screen_surf)
		dim = (96, 96)
		resized_imagdata = cv2.resize(imgdata, dim, interpolation = cv2.INTER_AREA)
		return resized_imagdata

	# Set the camera position to view the car from the top view
	def setCamera(self, car_x, car_y, car_a):
		eye_x , eye_y, eye_z = car_x, car_y, self.camera_height
		lookAt_x = car_x
		lookAt_y = car_y
		lookAt_z = 0
		gluLookAt(eye_x, eye_y, eye_z, lookAt_x, lookAt_y, lookAt_z, 0, 1, 0)

	# Display the stats (episode, speed, time elapsed) of the simulation on the display
	def draw_details(self, episode_no, speed, time_elapsed):
		font = pygame.font.SysFont('arial', 16)

		# Display episode number
		text = "          Episode: " + str(round(episode_no, 0))
		if episode_no == -1:
			text = "   VALIDATING MODEL NOW"
		textSurface = font.render(text, True, (255, 0, 0, 255)).convert_alpha()
		textData = pygame.image.tostring(textSurface, "RGBA", True)
		glWindowPos2d(220, 10)
		glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

		# Display speed
		text = "             Speed: " + str(round(speed, 1)) + " m/s"
		textSurface = font.render(text, True, (255, 0, 0, 255)).convert_alpha()
		textData = pygame.image.tostring(textSurface, "RGBA", True)
		glWindowPos2d(220, 40)
		glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

		# Display time elapsed
		text = "   Time Elapsed: " + str(round(time_elapsed, 1)) + " s"
		textSurface = font.render(text, True, (255, 0, 0, 255)).convert_alpha()
		textData = pygame.image.tostring(textSurface, "RGBA", True)
		glWindowPos2d(220, 70)
		glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

	# Draws path taken by the car (The higher the speed of the car, the further spaced the red dots)
	def draw_path(self):
		glColor3f(1, 0, 0)
		self.path.append((self.car_x + 0.1, self.car_y - 0.1 , 0.01))
		self.path.append((self.car_x + 0.1, self.car_y + 0.1, 0.01))
		self.path.append((self.car_x - 0.1, self.car_y + 0.1, 0.01))
		self.path.append((self.car_x - 0.1, self.car_y - 0.1, 0.01))
		
		glBegin(GL_QUADS)
		for i in range(len(self.path)):
			glVertex3fv(self.path[i])
		glEnd()

	# Draws the floor
	def draw_floor(self):
		glColor3f(1, 1, 1)
		glBegin(GL_QUADS)
		glVertex3fv((-100, -100, 0))
		glVertex3fv((100, -100, 0))
		glVertex3fv((100, 100, 0))
		glVertex3fv((-100, 100, 0))
		glEnd()

	# Draws the race track according to the track layout
	def draw_race_track(self, track_layout):
		glColor3f(0.5, 0.5, 0.5)
		glBegin(GL_QUAD_STRIP)
		for i in range(len(track_layout)):
			glVertex3fv((track_layout[i][0], track_layout[i][1], 0))
		glEnd()

	# Draws the race car according to its current centre coordinates
	def draw_race_car(self, x, y, a):
		glColor3f(0, 0, 0)
		glPushMatrix()
		glTranslatef(x, y, self.car_height/2)
		glRotatef(a, 0, 0, 1)
		glScalef(self.car_width/2, self.car_length/2, self.car_height/2)

		# vertices
		back_left_bot = (-1, -1, -1)
		back_left_top = (-1, -1, 1)
		back_right_bot = (1, -1, -1)
		back_right_top = (1, -1, 1)
		front_left_bot = (-1, 1, -1)
		front_left_top = (-1, 1, 1)
		front_right_bot = (1, 1, -1)
		front_right_top = (1, 1, 1)
		
		glBegin(GL_QUADS)
		# back surface
		glVertex3fv(back_left_bot)
		glVertex3fv(back_right_bot)
		glVertex3fv(back_right_top)
		glVertex3fv(back_left_top)

		# front surface
		glVertex3fv(front_right_bot)
		glVertex3fv(front_left_bot)
		glVertex3fv(front_left_top)
		glVertex3fv(front_right_top)

		# left surface
		glVertex3fv(front_left_bot)
		glVertex3fv(back_left_bot)
		glVertex3fv(back_left_top)
		glVertex3fv(front_left_top)

		# right surface
		glVertex3fv(front_right_bot)
		glVertex3fv(front_right_top)
		glVertex3fv(back_right_top)
		glVertex3fv(back_right_bot)

		# bot surface
		glVertex3fv(back_left_bot)
		glVertex3fv(front_left_bot)
		glVertex3fv(front_right_bot)
		glVertex3fv(back_right_bot)

		# top surface
		glVertex3fv(back_left_top)
		glVertex3fv(back_right_top)
		glVertex3fv(front_right_top)
		glVertex3fv(front_left_top)
		glEnd()
		glPopMatrix()