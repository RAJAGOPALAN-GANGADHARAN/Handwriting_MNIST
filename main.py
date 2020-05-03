from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color
from kivy.lang import Builder
#from model import predict
from kivy.core.window import Window
Window.size = (128, 128)

cnt = 0
class DrawInput(Widget):
    
    def on_touch_down(self, touch):
        #print(touch)
        with self.canvas:
            Color(0,0,0,1)
            touch.ud["line"] = Line(points=(touch.x, touch.y),width=10)
        
    def on_touch_move(self, touch):
        #print(touch)
        touch.ud["line"].points += (touch.x, touch.y)
		
    def on_touch_up(self, touch):
        pass

        #print("RELEASED!",touch)
kv_str = Builder.load_string("""
DrawInput:
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size
""")

class SimpleKivy4(App):
    
    def build(self):
        layout = FloatLayout(size=(150, 150))
        button = Button(text='Predict', size_hint=(.1, .1),
                        background_color=[1, 0, 0, 1])
        button.bind(on_press=self.callback)
        layout.add_widget(button)
        
        self.draw = kv_str
        layout.add_widget(self.draw)

        return layout
    def callback(self, event):
        global cnt
        self.draw.export_to_png("predict.jpg")
        cnt+=1

if __name__ == "__main__":
    SimpleKivy4().run()
