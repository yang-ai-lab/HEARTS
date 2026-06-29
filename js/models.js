// Model configuration — 1:1 port of src/config/models.ts
// ModelType:  'Reasoning' | 'Non-Reasoning' | 'Baseline'
// ModelSource:'Proprietary' | 'Open-source' | 'Reference'

// Target models to include in the analysis (column keys in merged_metric_results.csv)
export const TARGET_MODELS = [
  'claude_haiku',
  'deepseek',
  'gemini_2_5_flash',
  'gemini_2_5_pro',
  'gpt_4_1_mini',
  'gpt_5_mini',
  'grok_4_1_fast',
  'llama4',
  'nemontron_nano_12b',
  'qwen',
  'qwen_coder',
  'kimi',
  'glm_4_7',
  'glm_5',
  'minimax_m2',
  'gemini_3_1_pro',
  'naive',
];

// Model display names and metadata
export const MODELS = {
  claude_haiku:       { displayName: 'Claude 4.5 Haiku',       type: 'Non-Reasoning', source: 'Proprietary', icon: 'claude-color.png' },
  deepseek:           { displayName: 'DeepSeek V3.1',          type: 'Reasoning',     source: 'Open-source', icon: 'deepseek-color.png' },
  gemini_2_5_flash:   { displayName: 'Gemini 2.5 Flash',       type: 'Non-Reasoning', source: 'Proprietary', icon: 'gemini-color.png' },
  gemini_2_5_pro:     { displayName: 'Gemini 2.5 Pro',         type: 'Reasoning',     source: 'Proprietary', icon: 'gemini-color.png' },
  gemini_3_1_pro:     { displayName: 'Gemini 3.1 Pro',         type: 'Reasoning',     source: 'Proprietary', icon: 'gemini-color.png' },
  gpt_4_1_mini:       { displayName: 'GPT 4.1 Mini',           type: 'Non-Reasoning', source: 'Proprietary', icon: 'openai.png' },
  gpt_5_mini:         { displayName: 'GPT 5 Mini',             type: 'Non-Reasoning', source: 'Proprietary', icon: 'openai.png' },
  grok_4_1_fast:      { displayName: 'Grok 4.1 Fast',          type: 'Non-Reasoning', source: 'Proprietary', icon: 'grok.png' },
  llama4:             { displayName: 'Llama 4 Maverick',       type: 'Reasoning',     source: 'Open-source', icon: 'meta-color.png' },
  nemontron_nano_12b: { displayName: 'Nemotron Nano 12B V2',   type: 'Non-Reasoning', source: 'Open-source', icon: 'nvidia-color.png' },
  qwen:               { displayName: 'Qwen3 235B',             type: 'Reasoning',     source: 'Open-source', icon: 'qwen-color.png' },
  qwen_coder:         { displayName: 'Qwen3 Coder 480B',       type: 'Reasoning',     source: 'Open-source', icon: 'qwen-color.png' },
  kimi:               { displayName: 'Kimi K2 Thinking',       type: 'Reasoning',     source: 'Proprietary', icon: 'kimi-color.png' },
  glm_4_7:            { displayName: 'GLM 4.7',                type: 'Reasoning',     source: 'Open-source', icon: 'chatglm-color.png' },
  glm_5:              { displayName: 'GLM 5',                  type: 'Reasoning',     source: 'Open-source', icon: 'chatglm-color.png' },
  minimax_m2:         { displayName: 'MiniMax M2',             type: 'Reasoning',     source: 'Proprietary', icon: 'minimax-color.png' },
  naive:              { displayName: 'Naive Baseline',         type: 'Baseline',      source: 'Reference',   icon: 'naive.png' },

  // ---- Models additionally evaluated on the HEARTS-LIGHTENING track ----
  gemini_3_1_pro_preview: { displayName: 'Gemini 3.1 Pro',     type: 'Reasoning',     source: 'Proprietary', icon: 'gemini-color.png' },
  minimax_m2_5:       { displayName: 'MiniMax M2.5',           type: 'Reasoning',     source: 'Proprietary', icon: 'minimax-color.png' },
  moonshot_kimi_k2_5: { displayName: 'Kimi K2.5',              type: 'Reasoning',     source: 'Proprietary', icon: 'kimi-color.png' },
  gpt_5_4:            { displayName: 'GPT 5.4',                type: 'Reasoning',     source: 'Proprietary', icon: 'openai.png' },
  nemotron_super_3_120b: { displayName: 'Nemotron 3 Super 120B', type: 'Reasoning',  source: 'Open-source', icon: 'nvidia-color.png' },
};
