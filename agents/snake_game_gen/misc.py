from typing import List, Tuple, Union, Dict


class Slope(object):
    __slots__ = ('rise', 'run')
    def __init__(self, rise: int, run: int):
        self.rise = rise
        self.run = run

class Point(object):
    __slots__ = ('x', 'y')
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def copy(self) -> 'Point':
        x = self.x
        y = self.y
        return Point(x, y)

    def to_dict(self) -> Dict[str, int]:
        d = {}
        d['x'] = self.x
        d['y'] = self.y
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'Point':
        return Point(d['x'], d['y'])

    def __eq__(self, other: Union['Point', Tuple[int, int]]) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return other[0] == self.x and other[1] == self.y
        elif isinstance(other, Point) and self.x == other.x and self.y == other.y:
            return True
        return False

    def __sub__(self, other: Union['Point', Tuple[int, int]]) -> 'Point':
        if isinstance(other, tuple) and len(other) == 2:
            diff_x = self.x - other[0]
            diff_y = self.y - other[1]
            return Point(diff_x, diff_y)
        elif isinstance(other, Point):
            diff_x = self.x - other.x
            diff_y = self.y - other.y
            return Point(diff_x, diff_y)
        return None
    def __add__(self, other: Union['Point', Tuple[int, int]]) -> 'Point':
        if isinstance(other, tuple) and len(other) == 2:
            diff_x = self.x + other[0]
            diff_y = self.y + other[1]
            return Point(diff_x, diff_y)
        elif isinstance(other, Point):
            diff_x = self.x + other.x
            diff_y = self.y + other.y
            return Point(diff_x, diff_y)
        return None

    def __rsub__(self, other: Tuple[int, int]):
        diff_x = other[0] - self.x
        diff_y = other[1] - self.y
        return Point(diff_x, diff_y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self) -> str:
        return '({}, {})'.format(self.x, self.y)

def plotPixel(x1, y1, x2, y2, dx, dy, decide):
    
    
    
    
    pk = 2 * dy - dx
    points=[]

    

    
    
    
    
    for i in range(0, dx + 1):
        if decide:
            
            points.append((y1,x1))

        else:
            
            points.append((x1, y1))


        
        
        if (x1 < x2):
            x1 = x1 + 1
        else:
            x1 = x1 - 1
        if (pk < 0):

            
            
            if (decide == 0):

                
                pk = pk + 2 * dy
            else:

                
                
                pk = pk + 2 * dy
        else:
            if (y1 < y2):
                y1 = y1 + 1
            else:
                y1 = y1 - 1

            
            
            
            
            pk = pk + 2 * dy - 2 * dx
    return points




def findPoints(x1,y1,x2,y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    
    if (dx > dy):
        
        points=plotPixel(x1, y1, x2, y2, dx, dy, 0)

    
    else:
        
        points=plotPixel(y1, x1, y2, x2, dy, dx, 1)
    
    return points


VISION_16 = (

    Slope(-1, 0), Slope(-2, 1),  Slope(-1, 1),  Slope(-1, 2),

    Slope(0, 1),  Slope(1, 2),   Slope(1, 1),   Slope(2, 1),

    Slope(1, 0),  Slope(2, -1),  Slope(1, -1),  Slope(1, -2),

    Slope(0, -1), Slope(-1, -2), Slope(-1, -1), Slope(-2, -1)
)



VISION_8 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i%2==0])



VISION_4 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i%4==0])
