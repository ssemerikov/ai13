#lang racket
(define (getinversebit b n)
  (if (= 0 (bitwise-and b (arithmetic-shift 1 n))) 1 0)
  )


(define (bmp2vector fil)
  (define rest (make-vector 900)) ;вектор для файла 30х30
  (define f (open-input-file fil #:mode 'binary )) ;открываем файл для чтения в бинарном виде
  (define b1 0)
  (define b2 0)
  (define b3 0)
  (define b4 0)
  ;пропускаем голову BMP файла
  (do ((i 0 (+ 1 i)))
    ( (= i 62) )
    (set! b1 (read-byte f))
    )
  (do ((i 0 (+ 1 i))  (k 0))
    ( (= i 30) (close-input-port f) rest)
    ;считываем первые 4 байта
    (set! b1 (read-byte f))
    (set! b2 (read-byte f))
    (set! b3 (read-byte f))
    (set! b4 (read-byte f))
    ;И остачу
    (do ( (j 7 (- j 1)) )
      ( (= j -1) )
      (vector-set! rest k (getinversebit b1 j))
      (set! k (+ 1 k)) 
      )
    (do ( (j 7 (- j 1)) )
      ( (= j -1) )
      (vector-set! rest k (getinversebit b2 j))
      (set! k (+ 1 k))
      )
    (do ( (j 7 (- j 1)) )
      ( (= j -1) )
      (vector-set! rest k (getinversebit b3 j))
      (set! k (+ 1 k))
      )
    (do ( (j 7 (- j 1)) )
      ( (= j 1) )
      (vector-set! rest k (getinversebit b4 j))
      (set! k (+ 1 k))
      )
    )
  
  )

(define (bmp2vector16x16 fil)
  (define rest (make-vector 256)) ;тоже самое для 
  (define f (open-input-file fil #:mode 'binary ))
  (define b1 0)
  (define b2 0)
  (define b3 0)
  (define b4 0)
  ;skip BMP-header
  (do ((i 0 (+ 1 i)))
    ( (= i 62) )
    (set! b1 (read-byte f))
    )
  (do ((i 0 (+ 1 i))  (k 0))
    ( (= i 16) (close-input-port f) rest)
    (set! b1 (read-byte f))
    (set! b2 (read-byte f))
    (set! b3 (read-byte f))
    (set! b4 (read-byte f))
    (do ( (j 7 (- j 1)) )
      ( (= j -1) )
      (vector-set! rest k (getinversebit b1 j))
      (set! k (+ 1 k)) 
      )
    (do ( (j 7 (- j 1)) )
      ( (= j -1) )
      (vector-set! rest k (getinversebit b2 j))
      (set! k (+ 1 k))
      )
    
    )
  
  )



(define (1+ x) (+ 1 x))

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
(define (savematrix filename matr m n)
  (define f (open-output-file filename #:exists 'replace))
  (fprintf f "~S ~S \r\n" m n )
  (do ((i 0 (1+ i))) ((= i m))
    (do ((j 0 (1+ j))) ((= j n)) 
      (fprintf f "~S " (getmvalue matr i j))
      )
    (fprintf f "\r\n")
    
    )
  
  (close-output-port f)
  )



(define (generator)
  (define Xn (make-matrix 36 900))
  
  (do (( s 0 (+ s 1) ) (res 0)) ((= s 36))
    (set! res (bmp2vector (string-append (string(string-ref "0123456789abcdefghijklnmopqrstuvwxyz" s)) ".bmp")))
    (do (( j 0 (+ j 1) )) ((= j 900))
      (setmvalue Xn s j (vector-ref res j)) 
      )
    )
  
  (define Yn (make-matrix 36 36))  
  (do (( s 0 (+ s 1) )) ((= s 36))
    (setmvalue Yn s s 1)
    )
  
  (define Xt (make-matrix 6 900))
  
  (define Yt (make-matrix 6 36))
  
  (do (( s 0 (+ s 1) ) (zz (random 36) (random 36)) ) ((= s 5))
    (do (( j 0 (+ j 1) )  ) ((= j 900))
      (setmvalue Xt s j (getmvalue Xn zz j)) 
      )
    (do (( j 0 (+ j 1) )  ) ((= j 36))
      (setmvalue Yt s j (getmvalue Yn zz j)) 
      )
    )
  (define res (bmp2vector "smth.bmp"))
  (do (( j 0 (+ j 1) )) ((= j 900))
    (setmvalue Xt 5 j (vector-ref res j)) 
    )
  (setmvalue Yt 5 0 1)
  
  
  ;  
  ;  (do (( s 0 (+ s 1) )) ((= s 10))
  ;    (setmvalue Yt s 0 (bitwise-and (getmvalue Xt s 0) (bitwise-ior (getmvalue Xt s 1) (getmvalue Xt s 2)) ))
  ;    )
  ;
  ;Сохранение матрицы
  (savematrix "input.txt" Xn 36 900)
  (savematrix "test.txt" Xt 6 900)
  (savematrix "output.txt" Yn 36 36)
  (savematrix "output_for_test.txt" Yt 6 36)
  )

(generator)