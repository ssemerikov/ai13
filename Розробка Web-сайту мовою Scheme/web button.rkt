#lang web-server/insta

(struct post (title1 body1 ))
(define (can-parse-post? bindings)
  (and (exists-binding? 'title bindings)
       (exists-binding? 'body bindings)))

(define (parse-post bindings)
  (post (extract-binding/single 'title bindings)
        (extract-binding/single 'body bindings))
 (define f (open-input-file "output.txt" #:mode 'text))
  (with-output-to-file  
 f 
 (read)
  ))

(define (render-site-page a-site request)
  (response/xexpr
   `(html (head (title "My site"))
          (body
           (center(h1 "My site"))
           ,(render-posts a-site)
           (center(form
            (input ((name "title")))
            (input ((name "body")))
            (input ((type "submit")))))))))
 (define SITE
  (list (post "" "" )
        (post "" "" )))
(define (start request)
  (define a-site
    (cond [(can-parse-post? (request-bindings request))
           (cons (parse-post (request-bindings request))
                 
                 SITE)]
          [else
           SITE]))
  (render-site-page a-site request))

(define (render-post a-post)
  `(div ((class "post"))
        ,(post-title1 a-post)
        (p ,(post-body1 a-post))))
 

(define (render-posts a-site)
  `(div ((class "posts"))
        ,@(map render-post a-site)))