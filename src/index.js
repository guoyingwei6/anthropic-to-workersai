const DEFAULT_MODEL = '@cf/zai-org/glm-4.7-flash';

// 友好名称 → CF 模型 ID 映射
const MODEL_MAP = {
  // Meta Llama
  'llama-4-scout':   '@cf/meta/llama-4-scout-17b-16e-instruct',
  'llama-3.3-70b':   '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
  'llama-3.1-70b':   '@cf/meta/llama-3.1-70b-instruct',
  'llama-3.1-8b':    '@cf/meta/llama-3.1-8b-instruct',
  'llama-3.2-3b':    '@cf/meta/llama-3.2-3b-instruct',
  'llama-3.2-1b':    '@cf/meta/llama-3.2-1b-instruct',
  // OpenAI OSS
  'gpt-oss-120b':    '@cf/openai/gpt-oss-120b',
  'gpt-oss-20b':     '@cf/openai/gpt-oss-20b',
  // GLM
  'glm-4.7-flash':   '@cf/zai-org/glm-4.7-flash',
  // Qwen
  'qwen3-30b':       '@cf/qwen/qwen3-30b-a3b-fp8',
  'qwq-32b':         '@cf/qwen/qwq-32b',
  'qwen-coder-32b':  '@cf/qwen/qwen2.5-coder-32b-instruct',
  'qwen-72b':        '@cf/qwen/qwen2.5-72b-instruct',
  // DeepSeek
  'deepseek-r1':     '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
  'deepseek-r1-llama': '@cf/deepseek-ai/deepseek-r1-distill-llama-70b',
  // Mistral
  'mistral-small':   '@cf/mistralai/mistral-small-3.1-24b-instruct',
  'mistral-7b':      '@cf/mistral/mistral-7b-instruct-v0.1',
  // Google
  'gemma-3-12b':     '@cf/google/gemma-3-12b-it',
};

function resolveModel(reqModel) {
  if (!reqModel) return DEFAULT_MODEL;
  if (reqModel.startsWith('@cf/')) return reqModel;   // 直接传 CF 模型 ID
  return MODEL_MAP[reqModel] ?? DEFAULT_MODEL;
}

function isAuthorized(request, env) {
  // 没有配置 API_KEY 时跳过鉴权
  if (!env.API_KEY) return true;

  const apiKey = request.headers.get('x-api-key')
    ?? request.headers.get('authorization')?.replace(/^Bearer\s+/i, '');

  return apiKey === env.API_KEY;
}

export default {
  async fetch(request, env, ctx) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders() });
    }

    if (!isAuthorized(request, env)) {
      return jsonError('Unauthorized', 401);
    }

    const url = new URL(request.url);

    if (url.pathname === '/v1/messages' && request.method === 'POST') {
      return handleMessages(request, env, ctx);
    }

    if (url.pathname === '/v1/chat/completions' && request.method === 'POST') {
      return handleChatCompletions(request, env, ctx);
    }

    return jsonError('Not found', 404);
  },
};

async function handleMessages(request, env, ctx) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonError('Invalid JSON', 400);
  }

  const model = resolveModel(body.model);

  const messages = [];
  if (body.system) {
    const text = typeof body.system === 'string'
      ? body.system
      : body.system.filter(b => b.type === 'text').map(b => b.text).join('');
    messages.push({ role: 'system', content: text });
  }
  for (const msg of (body.messages || [])) {
    let content = msg.content;
    if (Array.isArray(content)) {
      content = content.filter(b => b.type === 'text').map(b => b.text).join('');
    }
    messages.push({ role: msg.role, content: content || '' });
  }

  const stream = body.stream !== false;
  const maxTokens = body.max_tokens || 4096;
  const msgId = 'msg_' + crypto.randomUUID().replace(/-/g, '').slice(0, 24);

  if (stream) {
    return handleStream(env, ctx, messages, maxTokens, msgId, model);
  } else {
    return handleNonStream(env, messages, maxTokens, msgId, model);
  }
}

async function handleStream(env, ctx, messages, maxTokens, msgId, model) {
  const aiStream = await env.AI.run(model, { messages, max_tokens: maxTokens, stream: true });

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();

  const send = async (event, data) => {
    await writer.write(enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
  };

  // 每 5 秒发一次 SSE 注释心跳，防止连接因空闲被断开
  const heartbeat = setInterval(async () => {
    try {
      await writer.write(enc.encode(': ping\n\n'));
    } catch {}
  }, 5000);

  ctx.waitUntil((async () => {
    try {
      await send('message_start', {
        type: 'message_start',
        message: {
          id: msgId, type: 'message', role: 'assistant', content: [],
          model, stop_reason: null, stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      });
      await send('content_block_start', {
        type: 'content_block_start', index: 0,
        content_block: { type: 'text', text: '' },
      });
      await send('ping', { type: 'ping' });

      const reader = aiStream.getReader();
      const dec = new TextDecoder();
      let outputTokens = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') continue;
          try {
            const parsed = JSON.parse(raw);
            const text = parsed.response ?? parsed.choices?.[0]?.delta?.content;
            if (text) {
              outputTokens++;
              await send('content_block_delta', {
                type: 'content_block_delta', index: 0,
                delta: { type: 'text_delta', text },
              });
            }
          } catch {}
        }
      }

      await send('content_block_stop', { type: 'content_block_stop', index: 0 });
      await send('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: outputTokens },
      });
      await send('message_stop', { type: 'message_stop' });
    } finally {
      clearInterval(heartbeat);
      await writer.close();
    }
  })());

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Access-Control-Allow-Origin': '*',
      'x-request-id': msgId,
    },
  });
}

async function handleNonStream(env, messages, maxTokens, msgId, model) {
  const result = await env.AI.run(model, { messages, max_tokens: maxTokens });
  const text = result.response || '';
  return jsonResponse({
    id: msgId, type: 'message', role: 'assistant',
    content: [{ type: 'text', text }],
    model,
    stop_reason: 'end_turn', stop_sequence: null,
    usage: {
      input_tokens: result.usage?.prompt_tokens || 0,
      output_tokens: result.usage?.completion_tokens || 0,
    },
  });
}

// ── OpenAI 兼容接口 /v1/chat/completions ──────────────────────────────────

async function handleChatCompletions(request, env, ctx) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonError('Invalid JSON', 400);
  }

  const model = resolveModel(body.model);
  const messages = (body.messages || []).map(msg => {
    let content = msg.content;
    if (Array.isArray(content)) {
      content = content.filter(b => b.type === 'text').map(b => b.text).join('');
    }
    return { role: msg.role, content: content || '' };
  });

  const stream = body.stream === true;
  const maxTokens = body.max_tokens || 4096;
  const chatId = 'chatcmpl-' + crypto.randomUUID().replace(/-/g, '').slice(0, 24);
  const created = Math.floor(Date.now() / 1000);

  if (stream) {
    return handleChatStream(env, ctx, messages, maxTokens, chatId, created, model);
  } else {
    return handleChatNonStream(env, messages, maxTokens, chatId, created, model);
  }
}

async function handleChatStream(env, ctx, messages, maxTokens, chatId, created, model) {
  const aiStream = await env.AI.run(model, { messages, max_tokens: maxTokens, stream: true });

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();

  const sendChunk = async (delta, finishReason = null) => {
    const chunk = {
      id: chatId, object: 'chat.completion.chunk', created, model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    };
    await writer.write(enc.encode(`data: ${JSON.stringify(chunk)}\n\n`));
  };

  const heartbeat = setInterval(async () => {
    try { await writer.write(enc.encode(': ping\n\n')); } catch {}
  }, 5000);

  ctx.waitUntil((async () => {
    try {
      await sendChunk({ role: 'assistant', content: '' });

      const reader = aiStream.getReader();
      const dec = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') continue;
          try {
            const parsed = JSON.parse(raw);
            const text = parsed.response ?? parsed.choices?.[0]?.delta?.content;
            if (text) await sendChunk({ content: text });
          } catch {}
        }
      }

      await sendChunk({}, 'stop');
      await writer.write(enc.encode('data: [DONE]\n\n'));
    } finally {
      clearInterval(heartbeat);
      await writer.close();
    }
  })());

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Access-Control-Allow-Origin': '*',
    },
  });
}

async function handleChatNonStream(env, messages, maxTokens, chatId, created, model) {
  const result = await env.AI.run(model, { messages, max_tokens: maxTokens });
  const text = result.response || '';
  return jsonResponse({
    id: chatId, object: 'chat.completion', created, model,
    choices: [{
      index: 0,
      message: { role: 'assistant', content: text },
      finish_reason: 'stop',
    }],
    usage: {
      prompt_tokens: result.usage?.prompt_tokens || 0,
      completion_tokens: result.usage?.completion_tokens || 0,
      total_tokens: (result.usage?.prompt_tokens || 0) + (result.usage?.completion_tokens || 0),
    },
  });
}

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key, anthropic-version',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
  };
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
  });
}

function jsonError(msg, status) {
  return jsonResponse({ error: { type: 'invalid_request_error', message: msg } }, status);
}
