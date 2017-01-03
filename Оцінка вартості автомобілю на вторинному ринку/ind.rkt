#lang racket
;трехлойная нейронная сеть (вход, скрытый слой, выход)

(define NUMIN 7)  ;размерность входа
(define NUMHID 15) ;размерность скрытого слоя
(define NUMPAT 30) ;количество шаблонов для обучения 
(define NUMOUT 1) ;размерность выхода

;матрицы весовых коэффициентов
(define WeightIH 0) ;соединения входа и скрытого слоя
(define WeightHO 0) ;соединения скрытого слоя и выходa

;для нормализации
(define mini 0)
(define maxi 0) 
(define mino 0)
(define maxo 0) 


;функции для создания векторов и матриц


(define (make-vector-my size)
  (if (and (integer? size) (> size 0)) 
      (make-vector size 0)
      (error "Недозволенная размерность вектора")
      )
  )


(define (make-matrix m n)
  (if (and (integer? m) (positive? m))
      (let ( (res (make-vector m 0)) )
        (do ((i 0 (+ i 1)))
          ( (= i m) res)
          (vector-set! res i (make-vector-my n))
          )
        )
      (error "Недозволенная размерность матрицы")
      )
  )

;
;;функции для получения/установки значений векторов и матриц
;
(define (getvvalue obj i)
  (vector-ref obj i)
  )

(define (getmvalue obj i j)
  (vector-ref (vector-ref obj i) j)
  )

(define (setvvalue vec i val)
  (vector-set! vec i val)
  )

(define (setmvalue matrix i j val)
  (define tempvec (vector-ref matrix i))
  (vector-set! tempvec j val)
  (vector-set! matrix i tempvec)
  )

(define (1+ x) (+ 1 x))
;
;;создание нейронной сети
;
(define (make-network _NUMIN _NUMOUT _NUMHID)
  (set! NUMIN _NUMIN)
  (set! NUMOUT _NUMOUT)
  (set! NUMHID _NUMHID)
  
  (set! WeightIH (make-matrix (1+ NUMIN) NUMHID))
  (set! WeightHO (make-matrix (1+ NUMHID) NUMOUT))
  
  (define smallwt 0.5)
  
  ;устанавливаем случайные весовые коэффициенты
  
  (do ((j 0 (+ 1 j)))
    ((= j NUMHID) )
    (do ((i 0 (+ i 1)))
      ( (= i (1+ NUMIN)))
      (setmvalue WeightIH i j  (* 2.0 (- (random) 0.5) smallwt))
      )
    )
  
  (do ((k 0 (+ 1 k)))
    ((= k NUMOUT) )
    (do ((j 0 (+ j 1)))
      ( (= j (1+ NUMHID)))
      (setmvalue WeightHO j k (* 2.0 (- (random) 0.5) smallwt))
      )
    )
  )
;
;
;
;;обучение нейронной сети
(define (train TrainInput TrainTarget Err MaxCount 
               DoOut)
  (let ((Error 0) (p 0) 
                  (eta 0.5) 
                  (alpha 0.9) 
                  (NUMPAT (vector-length TrainInput)) ;число обучающих шаблонов
                  (ranpat (make-vector-my NUMPAT))
                  (NumPattern NUMPAT)
                  (NumInput NUMIN)
                  (NumHidden NUMHID)
                  (NumOutput NUMOUT)
                  ;временные массивы
                  (DeltaWeightIH (make-matrix (1+ NUMIN) NUMHID))
                  (DeltaWeightHO (make-matrix (1+ NUMHID) NUMOUT))
                  (SumDOW (make-vector-my NUMHID))
                  (DeltaH (make-vector-my NUMHID))
                  (DeltaO (make-vector-my NUMOUT))
                  (SumH (make-vector-my NUMHID))
                  (Hidden (make-vector-my NUMHID))
                  (SumO (make-vector-my NUMOUT)) 
                  (Output (make-vector-my NUMOUT))
                  (Input (make-matrix NUMPAT NUMIN))
                  (Target (make-matrix NUMPAT NUMOUT))
                  )
    
    ;копируем тренировочные матрицы во временные во избежание порчи
    (do ((i 0 (1+ i))) ((= i NUMPAT))
      (do ((k 0 (1+ k))) ((= k NUMIN))
        (setmvalue Input i k (getmvalue TrainInput i k))
        )
      )
    (do ((i 0 (1+ i))) ((= i NUMPAT))
      (do ((k 0 (1+ k))) ((= k NUMOUT))
        (setmvalue Target i k (getmvalue TrainTarget i k))
        )
      )
    
    (set! mini (getmvalue Input 0 0))
    (set! maxi (getmvalue Input 0 0))
    (set! mino (getmvalue Target 0 0))
    (set! maxo (getmvalue Target 0 0))
    
    ;поиск граничных значений в числовых массивах
    (do ((i 0 (1+ i))) ((= i NumPattern))
      (do ((k 0 (1+ k))) ((= k NumInput))
        (when (> mini (getmvalue Input i k)) 
          (set! mini (getmvalue Input i k))
          )
        (when (< maxi (getmvalue Input i k)) 
          (set! maxi (getmvalue Input i k))
          )
        )
      (do ((k 0 (1+ k))) ((= k NumOutput))
        (when (> mino (getmvalue Target i k)) 
          (set! mino (getmvalue Target i k))
          )
        (when (< maxo (getmvalue Target i k)) 
          (set! maxo (getmvalue Target i k))
          )
        )
      )
    
    ;нормализация
    (do ((i 0 (1+ i))) ((= i NumPattern))
      (do ((k 0 (1+ k))) ((= k NumInput))
        (setmvalue Input i k 
                   (/ (- (getmvalue Input i k) mini) (- maxi mini))
                   )
        )
      (do ((k 0 (1+ k))) ((= k NumOutput))
        (setmvalue Target i k 
                   (/ (- (getmvalue Target i k) mino) (- maxo mino))
                   )
        )
      )  
    
    (set! Error (* 2 Err))              
    ;цикл обучения по достижению заданной ошибки или числа итераций
    (do ((epoch 0 (1+ epoch))) ((or (= epoch MaxCount) (< Error Err)) Error)
      ;перемешиваем шаблоны
      (do ((p 0 (1+ p))) ((= p NumPattern))
        (setvvalue ranpat p (random NumPattern))
        )
      (set! Error 0.0)
      
      ;цикл обучения по шаблонам
      (do ((np 0 (1+ np))) ((= np NumPattern))
        ;выбираем шаблон
        (set! p (getvvalue ranpat np))
        ;активация скрытого слоя
        (do ((j 0 (1+ j))) ((= j NumHidden))
          (setvvalue SumH j (getmvalue WeightIH 0 j))
          (do ((i 0 (1+ i))) ((= i NumInput))
            (setvvalue SumH j
                       (+ (getvvalue SumH j)
                          (* 
                           (getmvalue Input p i)
                           (getmvalue WeightIH (1+ i) j)
                           )
                          )
                       )
            )
          (setvvalue Hidden j (/ 1.0 
                                 (+ 
                                  1.0 
                                  (exp (- (getvvalue SumH j)))
                                  )    
                                 )
                     )
          )
        ;активация выходного слоя и вычисление ошибки
        (do ((k 0 (1+ k))) ((= k NumOutput))
          (setvvalue SumO k (getmvalue WeightHO 0 k))
          (do ((j 0 (1+ j))) ((= j NumHidden))
            (setvvalue SumO k (+ 
                               (getvvalue SumO k) 
                               (* 
                                (getvvalue Hidden j)
                                (getmvalue WeightHO (1+ j) k)
                                )
                               )
                       )
            )
          ;сигмоидальный вывод
          
          (setvvalue Output k (/ 1.0 
                                 (+ 
                                  1.0 
                                  (exp (- (getvvalue SumO k)))
                                  )    
                                 )
                     )
          (set! Error (+ 
                       Error
                       (*
                        0.5 
                        (- (getmvalue Target p k) (getvvalue Output k))
                        (- (getmvalue Target p k) (getvvalue Output k))
                        )
                       )
                )
          (setvvalue DeltaO k 
                     (*
                      (- (getmvalue Target p k) (getvvalue Output k))
                      (getvvalue Output k)
                      (- 1.0 (getvvalue Output k))
                      )
                     )
          )
        ;обратное распространение ошибки на скрытый слой
        (do ((j 0 (1+ j))) ((= j NumHidden))
          (setvvalue SumDOW j 0.0)
          (do ((k 0 (1+ k))) ((= k NumOutput))
            (setvvalue SumDOW j 
                       (+
                        (getvvalue SumDOW j)
                        (*
                         (getmvalue WeightHO (1+ j) k)
                         (getvvalue DeltaO k)
                         )
                        )
                       )
            )
          (setvvalue DeltaH j 
                     (* 
                      (getvvalue SumDOW j)
                      (getvvalue Hidden j)
                      (- 1.0 (getvvalue Hidden j))
                      )
                     )
          )
        (do ((j 0 (1+ j))) ((= j NumHidden))
          (setmvalue DeltaWeightIH 0 j 
                     (+
                      (* 
                       eta 
                       (getvvalue DeltaH j)
                       )
                      (*
                       alpha 
                       (getmvalue DeltaWeightIH 0 j)
                       )
                      )
                     )
          (setmvalue WeightIH 0 j 
                     (+
                      (getmvalue WeightIH 0 j)
                      (getmvalue DeltaWeightIH 0 j)
                      )
                     )
          (do ((i 0 (1+ i))) ((= i NumInput))
            (setmvalue DeltaWeightIH (1+ i) j 
                       (+
                        (* 
                         eta 
                         (getmvalue Input p i)
                         (getvvalue DeltaH j)
                         )
                        (*
                         alpha 
                         (getmvalue DeltaWeightIH (1+ i) j)
                         )
                        )
                       )
            (setmvalue WeightIH (1+ i) j 
                       (+
                        (getmvalue WeightIH (1+ i) j)
                        (getmvalue DeltaWeightIH (1+ i) j)
                        )
                       )
            )
          )
        (do ((k 0 (1+ k))) ((= k NumOutput))
          (setmvalue DeltaWeightHO 0 k
                     (+
                      (* 
                       eta 
                       (getvvalue DeltaO k)
                       )
                      (*
                       alpha 
                       (getmvalue DeltaWeightHO 0 k)
                       )
                      )
                     )
          (setmvalue WeightHO 0 k
                     (+
                      (getmvalue WeightHO 0 k)
                      (getmvalue DeltaWeightHO 0 k)
                      )
                     )
          (do ((j 0 (1+ j))) ((= j NumHidden))
            (setmvalue DeltaWeightHO (1+ j) k
                       (+
                        (* 
                         eta 
                         (getvvalue Hidden j)
                         (getvvalue DeltaO k)
                         )
                        (*
                         alpha 
                         (getmvalue DeltaWeightHO (1+ j) k)
                         )
                        )
                       )
            (setmvalue WeightHO (1+ j) k
                       (+
                        (getmvalue WeightHO (1+ j) k)
                        (getmvalue DeltaWeightHO (1+ j) k)
                        )
                       )
            )
          )
        )
      (when DoOut ;отладочный вывод
        (print (format "epoch=~a, error=~a" epoch Error))
        (newline)
        )
      
      )
    )
  )

;;подача сигнала на вход сети и получение результата
(define (getoutput BeInput) 
  (let ( 
        (Input (make-vector-my NUMIN))
        (Output (make-vector-my NUMOUT))
        (result (make-vector-my NUMOUT))
        (SumH (make-vector-my NUMHID))
        (Hidden (make-vector-my NUMHID))
        (SumO (make-vector-my NUMOUT))
        (NumInput NUMIN)
        (NumHidden NUMHID)
        (NumOutput NUMOUT)
        ) 
    ;нормализация входа
    (do ((k 0 (1+ k)))
      ((= k NumInput))
      (setvvalue Input k (/
                          (- (getvvalue BeInput k) mini)
                          (- maxi mini)
                          )
                 )
      )
    
    ;активация скрытого слоя
    (do ((j 0 (1+ j)))
      ((= j NumHidden))
      (setvvalue SumH j (getmvalue WeightIH 0 j))
      (do ((i 0 (1+ i)))
        ((= i NumInput))
        (setvvalue SumH j
                   (+ (getvvalue SumH j)
                      (* 
                       (getvvalue Input i)
                       (getmvalue WeightIH (1+ i) j)
                       )
                      )
                   )
        )
      (setvvalue Hidden j (/ 1.0 
                             (+ 
                              1.0 
                              (exp (- (getvvalue SumH j)))
                              )    
                             )
                 )
      )
    
    ;активация выходного слоя
    (do ((k 0 (1+ k)))
      ((= k NumOutput))
      (setvvalue SumO k (getmvalue WeightHO 0 k))
      (do ((j 0 (1+ j)))
        ((= j NumHidden))
        (setvvalue SumO k (+ 
                           (getvvalue SumO k) 
                           (* 
                            (getvvalue Hidden j)
                            (getmvalue WeightHO (1+ j) k)
                            )
                           )
                   )
        )
      (setvvalue Output k (/ 1.0 
                             (+ 
                              1.0 
                              (exp (- (getvvalue SumO k)))
                              )    
                             )
                 )
      )
    
    ;денормализация выхода
    (do ((k 0 (1+ k))) ( (= k NumOutput) result)
      (setvvalue result k (+
                           (*
                            (getvvalue Output k)
                            (- maxo mino)
                            )
                           mino
                           )
                 )
      )
    )
  )

;пример создания использования нейронной сети

;(define NUMPAT 60) ;число обучающих шаблонов - может переопределяться в файле
;(set! NUMIN  2)  ;размерность входа - может переопределяться в файле
;(set! NUMHID 13) ;число нейронов в скрытом слое
;(set! NUMOUT 1)  ;размерность выхода - может переопределяться в файле
;

(define (menu)
  (print "Выберите:")
  (newline)
  (print "1. Загрузка весовых коэффициентов")
  (newline)
  (print "2. Обучение сети (ДОЛГО!!!)")
  (newline)
  (print "3. Вычисления")
  (newline)
  (print "4. Выход")
  (newline)
  (define choice (read))
  (cond
    ((= choice 1) (loadcoeffs "wih.txt" "who.txt") (menu))
    ((= choice 2) (main) (menu))
    ((= choice 3) (compute) (menu))
    ((= choice 4) )
    (else (menu))
    )
  )


(define (readvector v)
  (do ((i 0 (1+ i)) )
    ((= i (vector-length v))   )
    (vector-set! v i (read))
    )
  )


(define (getletter v)
  (let ((num 0) (value (vector-ref v 0)))
    (do ( (i 1 (1+ i)))
      ((= i (vector-length v)))
      (when (< value (vector-ref v i))
        (set! num i)
        (set! value (vector-ref v i))
        )
      )
    (if (> value 0.5)
        (string(string-ref "abcdefghijklnmopqrstuvwxyz" num))
        "Unable recognize letter"
        )
    )
  )

(define (compute)
  (if (= 0 NUMHID)
      (print "Network wasn't loaded or created")
      (let ( (in (make-vector-my NUMIN)) (res 0) )
        (print "Input vector:")
        (readvector in)
        (set! res (getoutput in))
        (newline)
        (print " Полученный выход: ")
        (print res)
        (newline)
        (set! res (getletter res)) 
       ; (print " Recognized: ")
       ; (print res)
        (newline)
        )
      )
  )


(define (loadcoeffs first second)
  (set! WeightIH (loadmatrix first))
  (set! WeightHO (loadmatrix second))
  (set! NUMIN (- (vector-length WeightIH) 1))
  (set! NUMOUT (vector-length (vector-ref WeightHO 0) ))
  (set! NUMHID (vector-length (vector-ref WeightIH 0) ))
  )




(define (main)
  (define count 0)
  ;форматы файлов матриц: число_строк число_столбцов данные
  (define f (open-input-file "input.txt"))
  ;  (when (not f)
  ;    (progn
  ;      (princ "Ошибка открытия файла input.txt")
  ;      (return-from main nil)
  ;    )
  ;  )
  
  (set! NUMPAT (read f))
  (set! NUMIN (read f))
  
  (define Input (make-matrix NUMPAT NUMIN))
  
  (do ((i 0 (1+ i))) ((= i NUMPAT))
    (do ((k 0 (1+ k))) ((= k NUMIN))
      (setmvalue Input i k (read f))
      )
    )
  (close-input-port f)
  (set! f (open-input-file "output.txt"))
  ;  (when (not f)
  ;    (progn
  ;      (princ "Ошибка открытия файла output.txt")
  ;      (return-from main nil)
  ;    )
  ;  )
  (set! NUMPAT (read f))
  (set! NUMOUT (read f))
  
  (define Output (make-matrix NUMPAT NUMOUT))
  
  (do ((i 0 (1+ i))) ((= i NUMPAT))
    (do ((k 0 (1+ k))) ((= k NUMOUT))
      (setmvalue Output i k (read f))
      )
    )
  (close-input-port f)
  
  (make-network NUMIN NUMOUT NUMHID)
  (define in (make-vector-my NUMIN))
  (define res (make-vector-my NUMOUT))
  (define out (make-vector-my NUMOUT))
  
  (print
   (format "Размерность входа - ~a, размерность выхода - ~a, число шаблонов - ~a"
           NUMIN NUMOUT NUMPAT
           ))
  (newline)
  
  (train Input Output 0.001 900000 #t)
  (savematrix "wih.txt" WeightIH (+ NUMIN 1) NUMHID)
  (savematrix "who.txt" WeightHO (+ NUMHID 1) NUMOUT)
  
  (print "Исходные данные:")
  (newline)
  (do ((i 0 (1+ i))) ((= i NUMPAT))
    (print "Вход: ")
    (newline)
    (print (getvvalue Input i))
    (set! res (getoutput (getvvalue Input i)))
    (newline)
    (print " Эталонный выход: ")
    (newline)
    (print (getvvalue Output i))
    (print " Полученный выход: ")
    (newline)
    (print res)
    (newline)
    (when (zero? (remainder i 10))
      (print "Press Enter to continue")
      (read-char)
      
      )
    )
  
  (print "Тестовые данные:")
  (newline)
  (set! f (open-input-file "test.txt"))
  ;  (when (not f)
  ;    (progn
  ;      (print "Ошибка открытия файла test.txt")
  ;      (return-from main nil)
  ;    )
  ;  )
  (set! count (read f))
  (set! NUMIN (read f))
  (do ((i 0 (1+ i))) ((= i count))
    (do ((k 0 (1+ k))) ((= k NUMIN))
      (setvvalue in k (read f))
      )
    (print "Вход: ")
    (newline)
    (print in)
    (set! res (getoutput in))
    (newline)
    (print " Полученный выход: ")
    (newline)
    (print res)
    (newline)
    (when (zero? (remainder i 10))
      (print "Press Enter to continue")
      (read-char)
      )
    )
  
  (close-input-port f)

  )


;с помощью этих функций можно сохранять и 
;загружать матрицы весовых коэффициентов
;=====================================
;
(define (savematrix filename matr m n)
  (define f (open-output-file filename #:exists 'replace))
  (fprintf f "~S ~S " m n )
  (newline f)  
  (newline f)  
  (do ((i 0 (1+ i))) ((= i m))
    (do ((j 0 (1+ j))) ((= j n))
      (fprintf f "~S " (getmvalue matr i j))
      )
    (newline f)  
    )
  (newline f)  
  (fprintf f " ~S ~S ~S ~S " mini maxi mino maxo)
  (newline f)  
  (close-output-port f)
  )
;
;=====================================
;
(define (loadmatrix filename) 
  (define f (open-input-file filename))
  (define m (read f))
  (define n (read f))
  
  (define res (make-matrix m n ))
  (do ((i 0 (1+ i))) ((= i m))
    (do ((k 0 (1+ k))) ((= k n))
      (setmvalue res i k (read f))
      )
    )
  (set! mini (read f))
  (set! maxi (read f))
  (set! mino (read f))
  (set! maxo (read f))
  
  (close-input-port f)
  res
  )


(menu)