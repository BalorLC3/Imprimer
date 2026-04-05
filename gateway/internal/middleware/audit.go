package middleware

import (
	"context"
	"log"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type traceKey struct{}

// Audit is ISO 27001 audit middleware
// Every request that enters the system gets a trace ID here
// No request is ever processed without a trace ID
func Audit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		traceID := uuid.New().String()
		start := time.Now()
		ctx := context.WithValue(r.Context(), traceKey{}, traceID)

		r = r.WithContext(ctx)
		next.ServeHTTP(w, r)

		log.Printf("trace=%s method=%s path=%s duration=%s",
			traceID,
			r.Method,
			r.URL.Path,
			time.Since(start),
		)
	})
}

func TraceIDFrom(ctx context.Context) string {
	v, _ := ctx.Value(traceKey{}).(string)
	return v
}
