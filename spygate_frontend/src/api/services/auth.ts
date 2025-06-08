import { apiClient } from "../client";

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterData extends LoginCredentials {
  email: string;
  first_name?: string;
  last_name?: string;
}

export interface AuthTokens {
  access: string;
  refresh: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
}

export const authService = {
  // Login user
  login: async (credentials: LoginCredentials): Promise<AuthTokens> => {
    const response = await apiClient.post("/token/", credentials);
    const { access, refresh } = response.data;
    localStorage.setItem("token", access);
    localStorage.setItem("refreshToken", refresh);
    return response.data;
  },

  // Register new user
  register: async (data: RegisterData): Promise<User> => {
    const response = await apiClient.post("/auth/register/", data);
    return response.data;
  },

  // Logout user
  logout: () => {
    localStorage.removeItem("token");
    localStorage.removeItem("refreshToken");
  },

  // Get current user profile
  getCurrentUser: async (): Promise<User> => {
    const response = await apiClient.get("/auth/me/");
    return response.data;
  },

  // Update user profile
  updateProfile: async (data: Partial<User>): Promise<User> => {
    const response = await apiClient.patch("/auth/me/", data);
    return response.data;
  },

  // Refresh access token
  refreshToken: async (refresh: string): Promise<AuthTokens> => {
    const response = await apiClient.post("/token/refresh/", { refresh });
    const { access } = response.data;
    localStorage.setItem("token", access);
    return response.data;
  },

  // Verify token
  verifyToken: async (token: string): Promise<boolean> => {
    try {
      await apiClient.post("/token/verify/", { token });
      return true;
    } catch {
      return false;
    }
  },

  // Check if user is authenticated
  isAuthenticated: (): boolean => {
    return !!localStorage.getItem("token");
  },
};
