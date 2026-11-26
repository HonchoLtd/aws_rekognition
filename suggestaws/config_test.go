package suggestaws

import "testing"

func TestConfigFromEnv(t *testing.T) {
	for _, k := range []string{"AWS_REGION", "NOVA_LITE_MODEL_ID", "NOVA_MODEL_ID", "NOVA_DIM", "GUARDRAIL_ID", "GUARDRAIL_VERSION"} {
		t.Setenv(k, "")
	}
	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg != DefaultConfig() {
		t.Errorf("empty env = %+v, want defaults %+v", cfg, DefaultConfig())
	}
	if cfg.GuardrailID != "" || cfg.GuardrailVersion != "DRAFT" || cfg.Dimension != 1024 || cfg.LiteModelID != "us.amazon.nova-2-lite-v1:0" {
		t.Errorf("defaults = %+v", cfg)
	}

	t.Setenv("GUARDRAIL_ID", "gr-1")
	t.Setenv("GUARDRAIL_VERSION", "2")
	t.Setenv("NOVA_DIM", "3072")
	cfg, err = ConfigFromEnv()
	if err != nil || cfg.GuardrailID != "gr-1" || cfg.GuardrailVersion != "2" || cfg.Dimension != 3072 {
		t.Errorf("ConfigFromEnv = %+v, %v", cfg, err)
	}

	t.Setenv("NOVA_DIM", "1000")
	if _, err := ConfigFromEnv(); err == nil {
		t.Error("NOVA_DIM 1000 should be rejected")
	}
}
