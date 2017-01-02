#lang racket/gui

(require srfi/25)

(define (1+ n)
  (+ 1 n)
  )

;трехлойная нейронная сеть (вход, скрытый слой, выход

(define NUMIN 18)  ;размерность входа
(define NUMHID 14) ;размерность скрытого слоя
(define NUMOUT 14) ;размерность выхода

;матрицы весовых коэффициентов
(define WeightIH null) ;соединения входа и скрытого слоя
(define WeightHO null) ;соединения скрытого слоя и выходa

;для нормализации
(define mini 0)
(define maxi 0) 
(define mino 0)
(define maxo 0) 


;функции для создания векторов и матриц

(define (make-vector size)
  (if (and (integer? size) (positive? size)) 
      (make-array (shape 0 size) 0)
      (error "Недозволенная размерность вектора")
      )
  )

(define (make-matrix m n)
  (if (and (integer? m) (positive? m))
      (make-array (shape 0 m 0 n) 0)
      (error "Недозволенная размерность матрицы")
      )
  )

(define (vector-from-matrix m index)
  (define res (make-vector (array-end m 1)))
  (do [(i 0 (+ 1 i))] ((= i (array-end m 1)) res)
    (setvvalue res i (getmvalue m index i))
    )
  )

(define (print-vector v)
  (do [(i 0 (+ 1 i))] ((= i (array-end v 0)) (printf "\n"))
    (printf "~S " (getvvalue v i))
    )
  )
;функции для получения/установки значений векторов и матриц

(define (getvvalue obj i)
  (array-ref obj i)
  )


(define (getmvalue obj i j)
  (array-ref obj i j)
  )


(define (setvvalue vec i val)
  (array-set! vec i val)
  )

(define (setmvalue matrix i j val)
  (array-set! matrix i j val)
  )

(define (rando) 
  (/ (random 32000) 32000.0)
  )


;создание нейронной сети

(define (make-network _NUMIN _NUMOUT _NUMHID)
  (set! NUMIN _NUMIN)
  (set! NUMOUT _NUMOUT)
  (set! NUMHID _NUMHID)
  
  (set! WeightIH (make-matrix (+ 1 NUMIN) NUMHID))
  (set! WeightHO (make-matrix (+ 1 NUMHID) NUMOUT))
  
  (define smallwt 0.5)
  
  ;устанавливаем случайные весовые коэффициенты
  
  (do [(j 0 (+ 1 j))] ((= j NUMHID))
    (do [(i 0 (+ 1 i))] ((= i (+ 1 NUMIN)))
      (setmvalue WeightIH i j  (* 2.0 (- (rando) 0.5) smallwt))
      )
    )
  
  (do [(k 0 (+ 1 k))] ((= k NUMOUT))
    (do [(j 0 (+ 1 j))] ((= j (+ 1 NUMHID)))
      (setmvalue WeightHO j k (* 2.0 (- (rando) 0.5) smallwt))
      )
    )
  )



;обучение нейронной сети
(define (train TrainInput TrainTarget Err MaxCount DoOut)
  (let* (
         [Error (+ Err 1)]
         [eta 0.5]
         [alpha 0.9] 
         [NUMPAT (array-end TrainInput 0)] ;число обучающих шаблонов
         [ranpat (make-vector NUMPAT)]
         [NumPattern NUMPAT]
         [NumInput NUMIN]
         [NumHidden NUMHID]
         [NumOutput NUMOUT]
         ;временные массивы
         [DeltaWeightIH (make-matrix (1+ NUMIN) NUMHID)]
         [DeltaWeightHO (make-matrix (1+ NUMHID) NUMOUT)]
         [SumDOW (make-vector NUMHID)]
         [DeltaH (make-vector NUMHID)]
         [DeltaO (make-vector NUMOUT)]
         [SumH (make-vector NUMHID)]
         [Hidden (make-vector NUMHID)]
         [SumO (make-vector NUMOUT)]
         [Output (make-vector NUMOUT)]
         [Input (make-matrix NUMPAT NUMIN)]
         [Target (make-matrix NUMPAT NUMOUT)]
         )
    
    ;копируем тренировочные матрицы во временные во избежание порчи
    (do [(i 0 (+ 1 i))] ((= i NUMPAT))
      (do [(k 0 (+ 1 k))] ((= k NUMIN))
        (setmvalue Input i k (getmvalue TrainInput i k))
        )
      )
    (do [(i 0 (+ 1 i))] ((= i NUMPAT))
      (do [(k 0 (+ 1 k))] ((= k NUMOUT))        
        (setmvalue Target i k (getmvalue TrainTarget i k))
        )
      )
    
    (set! mini (getmvalue Input 0 0))
    (set! maxi (getmvalue Input 0 0))
    (set! mino (getmvalue Target 0 0))
    (set! maxo (getmvalue Target 0 0))
    
    ;поиск граничных значений в числовых массивах
    (do [(i 0 (+ 1 i))] ((= i NumPattern))
      (do [(k 0 (+ 1 k))] ((= k NumInput))        
        (when (> mini (getmvalue Input i k)) 
          (set! mini (getmvalue Input i k))
          )
        (when (< maxi (getmvalue Input i k)) 
          (set! maxi (getmvalue Input i k))
          )
        )
      (do [(k 0 (+ 1 k))] ((= k NumOutput))        
        (when (> mino (getmvalue Target i k)) 
          (set! mino (getmvalue Target i k))
          )
        (when (< maxo (getmvalue Target i k)) 
          (set! maxo (getmvalue Target i k))
          )
        )
      )
    
    ;нормализация
    (do [(i 0 (+ 1 i))] ((= i NumPattern))
      (do [(k 0 (+ 1 k))] ((= k NumInput))        
        (setmvalue Input i k 
                   (/ (- (getmvalue Input i k) mini) (- maxi mini))
                   )
        )
      (do [(k 0 (+ 1 k))] ((= k NumOutput))        
        (setmvalue Target i k 
                   (/ (- (getmvalue Target i k) mino) (- maxo mino))
                   )
        )
      )  
    
    ;цикл обучения по достижению заданной ошибки или числа итераций
    (do [(epoch 0 (+ 1 epoch))] ( (or (= epoch MaxCount) (< Error Err)) Error)        
      ;перемешиваем шаблоны
      (do [(p 0 (+ 1 p))] ((= p NumPattern))
        (setvvalue ranpat p (random NumPattern))
        )
      (set! Error 0.0)
      
      ;цикл обучения по шаблонам
      (do [(np 0 (+ 1 np)) (p (getvvalue ranpat 0) (getvvalue ranpat np))] ((= np NumPattern))            
        ;выбираем шаблон
        ;активация скрытого слоя
        (do [(j 0 (+ 1 j))] ((= j NumHidden))            
          (setvvalue SumH j (getmvalue WeightIH 0 j))
          (do [(i 0 (+ 1 i))] ((= i NumInput))
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
        (do [(k 0 (+ 1 k))] ((= k NumOutput))        
          (setvvalue SumO k (getmvalue WeightHO 0 k))
          (do [(j 0 (+ 1 j))] ((= j NumHidden))            
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
        (do [(j 0 (+ 1 j))] ((= j NumHidden))            
          (setvvalue SumDOW j 0.0)
          (do [(k 0 (+ 1 k))] ((= k NumOutput))        
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
        (do [(j 0 (+ 1 j))] ((= j NumHidden))            
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
          (do [(i 0 (+ 1 i))] ((= i NumInput))
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
        (do [(k 0 (+ 1 k))] ((= k NumOutput))        
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
          (do [(j 0 (+ 1 j))] ((= j NumHidden))            
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
      (when (and DoOut  (= 0 (remainder epoch 1000)));отладочный вывод
        (printf "epoch=~S, error=~S\n" epoch Error)
        )
      )
    )
  )

;подача сигнала на вход сети и получение результата
(define (getoutput BeInput 
                   ) 
  (let ( 
        [Input (make-vector NUMIN)]
        [Output (make-vector NUMOUT)]
        [result (make-vector NUMOUT)]
        [SumH (make-vector NUMHID)]
        [Hidden (make-vector NUMHID)]
        [SumO (make-vector NUMOUT)]
        [NumInput NUMIN]
        [NumHidden NUMHID]
        [NumOutput NUMOUT]
        )
    
    ;нормализация входа
    (do [(k 0 (+ 1 k))] ((= k NumInput))        
      (setvvalue Input k (/
                          (- (getvvalue BeInput k) mini)
                          (- maxi mini)
                          )
                 )
      )
    
    ;активация скрытого слоя
    (do [(j 0 (+ 1 j))] ((= j NumHidden))            
      (setvvalue SumH j (getmvalue WeightIH 0 j))
      (do [(i 0 (+ 1 i))] ((= i NumInput))
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
    (do [(k 0 (+ 1 k))] ((= k NumOutput))        
      (setvvalue SumO k (getmvalue WeightHO 0 k))
      (do [(j 0 (+ 1 j))] ((= j NumHidden))            
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
    (do [(k 0 (+ 1 k))] ((= k NumOutput) result)        
      (setvvalue result k (+
                           (*
                            (getvvalue Output k)
                            (- maxo mino)
                            )
                           mino
                           )
                 )
      )
    ))

;пример создания использования нейронной сети

(define NUMPAT 60) ;число обучающих шаблонов - может переопределяться в файле
(set! NUMIN  18)  ;размерность входа - может переопределяться в файле
(set! NUMHID 14) ;число нейронов в скрытом слое
(set! NUMOUT 14)  ;размерность выхода - может переопределяться в файле

(define (main)
  
  ;форматы файлов матриц: число_строк число_столбцов данные
  (define f (open-input-file "input.txt"))
  
  (set! NUMPAT (read f))
  (set! NUMIN (read f))
  
  (define Input (make-matrix NUMPAT NUMIN))
  
  (do [(i 0 (+ 1 i))] ((= i NUMPAT))
    (do [(k 0 (+ 1 k))] ((= k NUMIN))        
      (setmvalue Input i k (read f))
      )
    )
  (close-input-port f)
  
  (set! f (open-input-file "output.txt"))
  
  (set! NUMPAT (read f))
  (set! NUMOUT (read f))
  
  (define Output (make-matrix NUMPAT NUMOUT))
  
  (do [(i 0 (+ 1 i))] ((= i NUMPAT))
    (do [(k 0 (+ 1 k))] ((= k NUMOUT))        
      (setmvalue Output i k (read f))
      )
    )
  (close-input-port f)
  
  (make-network NUMIN NUMOUT NUMHID)
  (define in (make-vector NUMIN))
  (define res (make-vector NUMOUT))
  (define out (make-vector NUMOUT))
  
  (printf
   "Размерность входа - ~S, размерность выхода - ~S, число шаблонов - ~S\n"
   NUMIN NUMOUT NUMPAT
   )
  
  (train Input Output 0.000001 150000 #t)
  (savematrix "wih.txt" WeightIH (+ NUMIN 1) NUMHID)
  (savematrix "who.txt" WeightHO (+ NUMHID 1) NUMOUT)
  
  (display "Исходные данные:\n")
  (do [(i 0 (+ 1 i)) (res null)] ((= i NUMPAT))
    (display "Вход: ")
    (print-vector (vector-from-matrix Input i))
    (set! res (getoutput (vector-from-matrix Input i)))
    (display " Эталонный выход: ")
    (print-vector (vector-from-matrix Output i))
    (printf " Полученный выход: ")
    (print-vector res)
    (when (zero? (remainder i 10))
      (display "Press Enter to continue")
      (read-char)
      )
    )
  
  (display "Тестовые данные:\n")
  (set! f (open-input-file "test.txt"))
  
  (define count (read f))
  (set! NUMIN (read f))
  (do [(i 0 (+ 1 i)) (res null)] ((= i count))
    (do [(k 0 (+ 1 k))] ((= k NUMIN))        
      (setvvalue in k (read f))
      )
    (display "Вход: ")
    (print-vector in)
    (set! res (getoutput in))
    (display " Полученный выход: ")
    (print-vector res)
    (when (zero? (remainder i 10))
      (display "Press Enter to continue")
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
  (define f (open-output-file filename  #:exists 'replace))
  
  (fprintf f "~S ~S " m n )
  (do [(i 0 (+ 1 i))] ((= i m))
    (do [(j 0 (+ 1 j))] ((= j n))            
      (fprintf f "~S " (getmvalue matr i j))
      )
    ;(format f "~S" (code-char 13))  
    )
  (fprintf f " ~S ~S " mini maxi)
  ;(format f "~S" (code-char 13))  
  (fprintf f " ~S ~S " mino maxo)
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
  
  (do [(i 0 (+ 1 i))] ((= i m))
    (do [(k 0 (+ 1 k))] ((= k n))        
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

(define (loadcoeffs first second)
  (set! WeightIH (loadmatrix first))
  (set! WeightHO (loadmatrix second))
  (set! NUMIN (- (array-end WeightIH 0) 1))
  (set! NUMOUT (array-end  WeightHO 1))
  (set! NUMHID (array-end WeightIH 1))
  )

(define (readvector v)
  (do ((i 0 (1+ i)) )
    ((= i (array-end v 0))   )
    (array-set! v i (read))
    )
  )

(define список-симптомов '(
"Высокая температура?(больше 38)"
"Средняя температура?(между 37,2 и 37,9)"
"Сухой кашель?"
"Мокрий кашель?"
"Хрипы при дыхании?"
"Появляется ли отдышка?"
"Вас знобит?"
"Побледнела ли ваша кожа?"
"Покраснела ли ваша кожа?"
"Есть ли у вас тахикардия?"
"У вас высокое давление?(свыше 140/90)"
"У вас низкое давление?(ниже 110/70)"
"Боль в области живота?"
"Боль в области грудной клетки?"
"Головная боль?"
"Чувствуете ли вы затруднение дыхания?"
"Чувствуете ли вы увеличение грудной клетки?"
"Присутствует ли у вас кровотечение или кровохаркание?"
                          )
  )


(define (readdata v)
  (display "Опрос пациента:\n")
  (display "Введите 1, если симптом присутствует, и 0, если нет: ")
  (newline)
  (do ((i 0 (1+ i)) )
    ((= i (array-end v 0))   v)
    (display (list-ref список-симптомов i))
    (newline)
    (array-set! v i (read))
    )
     )


(define список-болезней '(
"Абсцесс легкого"
"Бронхиальная астма"
"Бронхоэктазия"
"Острый бронхит"
"Кровохарканье и легочное кровотечение"
"Хронический бронхит"
"Эмфизема легких"
"Плеврит"
"Пневмоконнозы"
"Острые пневмонии"
"Пневмосклероз"
"Пневмоторакс"
"Рак легкого"
"Саркоидоз легких"


                          ))

(define (болезнь res)
  (let ([max (array-ref res 0)] [index 0])
    (do ((i 0 (1+ i)) )
      ((= i (array-end res 0)) (format "Диагноз: ~A (с вероятностью ~A)\n Будьте здоровы!\n" (list-ref список-болезней index) max ))
      (when (< max (array-ref res i))
        (set! max (array-ref res i))
        (set! index i)
        )
      )
    )
  )

(define (compute)
  (if (= 0 NUMHID)
      (display "Network wasn't loaded or created")
      (let ( (in (make-vector NUMIN)) (res 0) )
        ;(display "Input array:")
        ;(readvector in)
        (readdata in)
        (set! res (getoutput in))
        (newline)
        ;(display " Полученный выход: ")
        ;(print res)
        ;(print-vector res)
        (display (болезнь res))
        (newline)
        )
      )
  )

(define (menu)
  (display "Выберите:")
  (newline)
  (display "1. Обучение сети")
  (newline)
  (display "2. Определить диагноз по симптомам")
  (newline)
  (display "3. Выход")
  (newline)
  (define choice (read))
  (cond
    ((= choice 1) (main))
    ((= choice 2) (loadcoeffs "wih.txt" "who.txt") (compute) (menu))
    ((= choice 3) )
    (else (menu))
    )
  )

(menu)