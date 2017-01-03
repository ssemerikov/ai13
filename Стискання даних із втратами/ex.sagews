︠1de8a1cc-b467-4dec-a0a8-ee9397d0e1fcs︠
from __future__ import print_function
import sys
import re


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args)

    def done(self):
        self.current = self.total
        self()
        print('')

import cv2
@interact
def fourier_on_stock(N=slider(0, 200, 1, 100, "Кількість гармонік"), name=selector(["photo3.png", "photo.png", "lines.png"], "Назва зображення", "photo3.png", 1, 1, 70, False, None)):
    im = cv2.imread(name)
    width=len(im[0])
    height=len(im)
    show("Зображення завантажено")
    show("Розміри:", width, "*", height)
    izmeren=[]
    for i in range(height):
        for j in range(width):
            for k in range(3):
                izmeren.append(im[i][j][k])
    show("Зображення перетворено у матрицю")
    show("Створюємо матриці Х та Y")
    l=len(izmeren)-1
    Y=matrix(l+1,1,izmeren)
    X=matrix(RR,l+1,N+1)
    show("Заповнюємо матрицю Х")
    pr2=((l+1)*(N+1))/100
    progress = ProgressBar((l+1)*(N+1), fmt=ProgressBar.FULL)
    z=0
    for i in range(0,l+1,1):
        for j in range(0,N+1,1):
            X[i,j]=cos(pi*j*i/l)
            if(progress.current<=z):
                    progress.current += pr2
                    progress()
            z+=1
    progress.done()
    show("Обчислюємо матриці А та С для виведення функції")
    A=(X.transpose()*X)^(-1)*X.transpose()*Y
    var('n')
    C=matrix(1,N+1,[cos(pi*n*x/l) for n in range(0,N+1,1)])
    show("Обчислюємо гармоніки")
    F(x)=(C*A)[0,0]
    show("Виведені гармоніки")
    html("$f(x) $=$%s$"%latex(F(x)))
    show("Розраховуємо нові дані зображення")
    im2 = cv2.imread(name)
    z=0
    pr=(height*width*3)/100
    progress = ProgressBar(height*width*3, fmt=ProgressBar.FULL)
    for i in range(height):
        for j in range(width):
            for k in range(3):
                im2[i][j][k]=F(z)
                z=z+1
                if(progress.current<=z-1):
                    progress.current += pr
                    progress()
    progress.done()
    show("Починається запис зображення")
    cv2.imwrite("photo2.png",im2)
    show("Зображення записано")
    koef=(l+1)/(N*8+2)
    show("Коефіцієнт стискання =", koef)
    html("<img src=./"+name+" height=30% width=30%>")
    html("<img src=./photo2.png height=30% width=30%>")
︡e37c2157-5e7c-4e4c-8b05-16bcfe62af88︡{"interact":{"controls":[{"animate":true,"control_type":"slider","default":100,"display_value":true,"label":"Кількість гармонік","vals":["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92","93","94","95","96","97","98","99","100","101","102","103","104","105","106","107","108","109","110","111","112","113","114","115","116","117","118","119","120","121","122","123","124","125","126","127","128","129","130","131","132","133","134","135","136","137","138","139","140","141","142","143","144","145","146","147","148","149","150","151","152","153","154","155","156","157","158","159","160","161","162","163","164","165","166","167","168","169","170","171","172","173","174","175","176","177","178","179","180","181","182","183","184","185","186","187","188","189","190","191","192","193","194","195","196","197","198","199","200"],"var":"N","width":null},{"button_classes":null,"buttons":false,"control_type":"selector","default":0,"label":"Назва зображення","lbls":["photo3.png","photo.png","lines.png"],"ncols":1,"nrows":1,"var":"name","width":70}],"flicker":false,"id":"640ecae1-4190-40f0-80f6-c15e6071ae0d","layout":[[["N",12,null]],[["name",12,null]],[["",12,null]]],"style":"None"}}︡{"done":true}︡









