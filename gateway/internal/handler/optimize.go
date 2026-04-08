package handler

import (
	"net/http"

	gen "github.com/BalorLC3/Imprimer/gateway/gen"
	"github.com/BalorLC3/Imprimer/gateway/internal/client"
	"github.com/BalorLC3/Imprimer/gateway/internal/httpx"
)

// Handles POST /optimize
type OptimizeHandler struct {
	engine *client.PythonClient
}

func NewOptimizeHandler(engine *client.PythonClient) *OptimizeHandler {
	return &OptimizeHandler{engine: engine}
}

// same pattern as to encode later
type optimizeRequest struct {
	Task           string `json:"task"`
	BasePrompt     string `json:"base_prompt"`
	InputExample   string `json:"input_example"`
	ExpectedOutput string `json:"expected_output"`
	NTrials        int32  `json:"n_trials"`
}

type optimizeResponse struct {
	BestPrompt string  `json:"best_prompt"`
	BestScore  float32 `json:"best_score"`
	Trials     int32   `json:"trials"`
}

func (h *OptimizeHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req optimizeRequest
	if !httpx.DecodeJSON(w, r, &req) {
		return
	}

	// Validate all fields and NTrials being strictly positive and more than zero
	if req.Task == "" ||
		req.BasePrompt == "" ||
		req.InputExample == "" ||
		req.ExpectedOutput == "" ||
		req.NTrials <= 0 {

		httpx.WriteError(w, http.StatusBadRequest, "all fields must be filled")
		return
	}

	grpcResp, err := h.engine.Optimize(r.Context(), &gen.OptimizeRequest{
		Task:           req.Task,
		BasePrompt:     req.BasePrompt,
		InputExample:   req.InputExample,
		ExpectedOutput: req.ExpectedOutput,
		NTrials:        req.NTrials,
	})

	if err != nil {
		http.Error(w, "engine error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp := optimizeResponse{
		BestPrompt: grpcResp.BestPrompt,
		BestScore:  grpcResp.BestScore,
		Trials:     grpcResp.Trials,
	}

	// serialize
	httpx.WriteJSON(w, http.StatusOK, resp)
}
