(function () {
  'use strict';

  const BASE = (window.FINORAQI_API || '') + '/api/v1';
  const TOKEN_KEY = 'finoraqi_v2_token';
  const USER_KEY = 'finoraqi_v2_user';

  class ApiError extends Error {
    constructor(status, detail) {
      super(typeof detail === 'string' ? detail : formatDetail(detail));
      this.status = status;
      this.detail = detail;
    }
  }

  function formatDetail(detail) {
    if (Array.isArray(detail)) {
      return detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
    }
    return detail ? JSON.stringify(detail) : 'Erro inesperado';
  }

  const session = {
    get token() { return localStorage.getItem(TOKEN_KEY); },
    get user() {
      try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null'); } catch { return null; }
    },
    save(data) {
      localStorage.setItem(TOKEN_KEY, data.token);
      localStorage.setItem(USER_KEY, JSON.stringify(data.user));
    },
    clear() {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    }
  };

  async function request(method, path, body, opts = {}) {
    const headers = {};
    if (session.token) headers.Authorization = `Bearer ${session.token}`;
    let payload;
    if (body instanceof FormData) {
      payload = body;
    } else if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      payload = JSON.stringify(body);
    }
    const res = await fetch(BASE + path, { method, headers, body: payload });
    if (res.status === 401 && !opts.allowUnauthorized) {
      session.clear();
      window.dispatchEvent(new CustomEvent('finoraqi:logout'));
    }
    if (res.status === 204) return null;
    const type = res.headers.get('content-type') || '';
    const data = type.includes('application/json') ? await res.json() : await res.blob();
    if (!res.ok) throw new ApiError(res.status, data && data.detail !== undefined ? data.detail : data);
    return data;
  }

  const get = (p) => request('GET', p);
  const post = (p, b) => request('POST', p, b);
  const put = (p, b) => request('PUT', p, b);
  const del = (p) => request('DELETE', p);

  function form(fields) {
    const fd = new FormData();
    Object.entries(fields).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== '') fd.append(k, v); });
    return fd;
  }

  window.FinoraQi = {
    ApiError,
    session,
    health: () => fetch((window.FINORAQI_API || '') + '/health').then((r) => r.json()),
    auth: {
      async register(nome, email, senha) {
        const data = await request('POST', '/auth/register', { nome, email, senha }, { allowUnauthorized: true });
        session.save(data);
        return data;
      },
      async login(email, senha) {
        const data = await request('POST', '/auth/login', { email, senha }, { allowUnauthorized: true });
        session.save(data);
        return data;
      },
      logout: () => session.clear(),
      profile: () => get('/profile')
    },
    finance: {
      summary: () => get('/dashboard/summary'),
      transactions: () => get('/transactions'),
      createTransaction: (tx) => post('/transactions', tx),
      deleteTransaction: (id) => del(`/transactions/${id}`)
    },
    ai: {
      categories: () => get('/ai/categories'),
      categorize: (descriptions) => post('/ai/categorize', { descriptions }),
      feedback: (description, category) => post('/ai/categorize/feedback', { description, category }),
      insights: (horizon = 30) => get(`/ai/insights?horizon=${horizon}`)
    },
    payments: {
      process: (p) => post('/payments/process', p),
      list: (limit = 20) => get(`/payments?limit=${limit}`),
      benchmark: (n = 10000) => get(`/payments/benchmark?n=${n}`)
    },
    score: {
      operation: (op) => post('/credit-operation', op),
      get: (cpf) => get(`/score/${encodeURIComponent(cpf)}`)
    },
    data: {
      profile: (file) => post('/data/profile', form({ file })),
      clean: (file) => post('/data/clean', form({ file }))
    },
    automl: {
      analyze: (file) => post('/automl/analyze', form({ file })),
      train: (file, target, problemType) => post('/automl/train', form({ file, target, problem_type: problemType })),
      models: () => get('/automl/models'),
      predict: (modelId, rows) => post('/automl/predict', { model_id: modelId, rows })
    },
    flags: {
      list: () => get('/flags'),
      set: (name, enabled) => put(`/flags/${encodeURIComponent(name)}`, { enabled })
    },
    cashpool: {
      pools: () => get('/cashpool/pools'),
      create: (pool) => post('/cashpool/pools', pool),
      positions: (id) => get(`/cashpool/pools/${id}/positions`),
      importPositions: (id, positions) => post(`/cashpool/pools/${id}/positions`, { positions }),
      sweep: (id, dryRun) => post(`/cashpool/pools/${id}/sweep?dry_run=${dryRun ? 'true' : 'false'}`),
      obligations: (id, obligations) => post(`/cashpool/pools/${id}/obligations`, { obligations }),
      netting: (id, cycleDate, dryRun) => post(`/cashpool/pools/${id}/netting`, { cycle_date: cycleDate, dry_run: !!dryRun }),
      yieldAlloc: (id, body) => post(`/cashpool/pools/${id}/yield`, body),
      history: (id) => get(`/cashpool/pools/${id}/history`)
    }
  };
})();
