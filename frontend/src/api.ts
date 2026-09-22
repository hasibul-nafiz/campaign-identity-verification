export const API_BASE =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8001";

export type Pose = "front" | "left" | "right" | "up" | "down" | "unknown";

export interface ImageStatus {
  index: number;
  filename: string | null;
  ok: boolean;
  pose: Pose | null;
  error: string | null;
  replaced_earlier_image?: boolean;
}

export interface RegisterResponse {
  person: { id: number; name: string };
  images: ImageStatus[];
  templates_written: number;
  poses: string[];
}

/** The 400 body for a missing front pose still carries per-image rows. */
export interface RegisterError {
  detail: string;
  images?: ImageStatus[];
}

export interface Person {
  id: number;
  name: string;
  created_at: string;
  poses: string[];
  template_count: number;
}

export interface CampaignColor {
  id: number;
  hex: string;
  label: string;
}

export interface BadgeRefSummary {
  id: number;
  width: number;
  height: number;
  keypoints: number;
  url: string;
}

export interface Campaign {
  id: number;
  name: string;
  active: boolean;
  created_at: string;
  colors: CampaignColor[];
  badge_ref_count: number;
  overrides: { badge_min_inliers: number | null; shirt_delta_e_max: number | null };
  effective: { badge_min_inliers: number; shirt_delta_e_max: number };
  badge_refs?: BadgeRefSummary[];
}

export interface UploadRow {
  filename: string | null;
  ok: boolean;
  error: string | null;
  keypoints: number;
  keypoints_in_upload?: number;
  cropped?: boolean;
  id: number | null;
}

export interface DetectResponse {
  passed: boolean;
  person: { id: number; name: string; score: number; matched_pose: string } | null;
  face: { box: [number, number, number, number]; score: number; pose: Pose; yaw: number; pitch: number };
  campaign: { id: number; name: string; auto: boolean } | null;
  campaign_candidates?: { id: number; name: string; inliers: number; ok: boolean }[];
  badge: {
    ok: boolean;
    inliers: number;
    matches: number;
    reason: string;
    box: [number, number][] | null;
    matched_ref_id: number | null;
  } | null;
  shirt: {
    ok: boolean;
    delta_e: number;
    rgb: [number, number, number];
    expected_rgb: [number, number, number];
    expected_hex: string;
    matched_color: string;
    coverage: number;
  } | null;
  image_size: [number, number];
  torso_box: [number, number, number, number] | null;
  badge_crop_b64: string | null;
  best_score: number | null;
  reason?: string;
  timings_ms: Record<string, number>;
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly body?: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let body: unknown;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : `Request failed (${res.status})`;
    throw new ApiError(detail, res.status, body);
  }
  return body as T;
}

function failedToReach(err: unknown): never {
  if (err instanceof ApiError) throw err;
  throw new ApiError(`Cannot reach the API at ${API_BASE}. Is it running?`, 0);
}

export async function registerPerson(
  name: string,
  images: Blob[],
): Promise<RegisterResponse> {
  const form = new FormData();
  form.append("name", name);
  images.forEach((blob, i) => form.append("images", blob, `pose_${i}.jpg`));
  try {
    const res = await fetch(`${API_BASE}/register`, { method: "POST", body: form });
    return await parse<RegisterResponse>(res);
  } catch (err) {
    return failedToReach(err);
  }
}

export async function detect(
  image: Blob,
  campaignId?: number,
  expectedColor?: string,
): Promise<DetectResponse> {
  const form = new FormData();
  form.append("image", image, "frame.jpg");
  if (campaignId !== undefined) form.append("campaign_id", String(campaignId));
  if (expectedColor) form.append("expected_color", expectedColor);
  try {
    const res = await fetch(`${API_BASE}/detect`, { method: "POST", body: form });
    return await parse<DetectResponse>(res);
  } catch (err) {
    return failedToReach(err);
  }
}

export async function listPeople(): Promise<Person[]> {
  try {
    const res = await fetch(`${API_BASE}/people`);
    const body = await parse<{ people: Person[] }>(res);
    return body.people;
  } catch (err) {
    return failedToReach(err);
  }
}

export async function deletePerson(id: number): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/people/${id}`, { method: "DELETE" });
    await parse<{ deleted: number }>(res);
  } catch (err) {
    return failedToReach(err);
  }
}

export async function listCampaigns(): Promise<Campaign[]> {
  try {
    const res = await fetch(`${API_BASE}/campaigns`);
    return (await parse<{ campaigns: Campaign[] }>(res)).campaigns;
  } catch (err) {
    return failedToReach(err);
  }
}

export async function getCampaign(id: number): Promise<Campaign> {
  try {
    return await parse<Campaign>(await fetch(`${API_BASE}/campaigns/${id}`));
  } catch (err) {
    return failedToReach(err);
  }
}

export async function createCampaign(
  name: string,
  colors: string[],
  images: Blob[],
): Promise<Campaign & { uploads: UploadRow[] }> {
  const form = new FormData();
  form.append("name", name);
  colors.forEach((c) => form.append("colors", c));
  images.forEach((b, i) => form.append("images", b, `ref_${i}.png`));
  try {
    const res = await fetch(`${API_BASE}/campaigns`, { method: "POST", body: form });
    return await parse<Campaign & { uploads: UploadRow[] }>(res);
  } catch (err) {
    return failedToReach(err);
  }
}

export async function updateCampaign(
  id: number,
  fields: { name?: string; active?: boolean; colors?: string[];
            badge_min_inliers?: number | ""; shirt_delta_e_max?: number | "" },
): Promise<Campaign> {
  const form = new FormData();
  if (fields.name !== undefined) form.append("name", fields.name);
  if (fields.active !== undefined) form.append("active", String(fields.active));
  if (fields.colors !== undefined) fields.colors.forEach((c) => form.append("colors", c));
  if (fields.badge_min_inliers !== undefined && fields.badge_min_inliers !== "")
    form.append("badge_min_inliers", String(fields.badge_min_inliers));
  if (fields.shirt_delta_e_max !== undefined && fields.shirt_delta_e_max !== "")
    form.append("shirt_delta_e_max", String(fields.shirt_delta_e_max));
  try {
    const res = await fetch(`${API_BASE}/campaigns/${id}`, { method: "PATCH", body: form });
    return await parse<Campaign>(res);
  } catch (err) {
    return failedToReach(err);
  }
}

export async function deleteCampaign(id: number): Promise<void> {
  try {
    await parse(await fetch(`${API_BASE}/campaigns/${id}`, { method: "DELETE" }));
  } catch (err) {
    return failedToReach(err);
  }
}

export async function addBadgeRefs(
  id: number,
  images: Blob[],
): Promise<Campaign & { uploads: UploadRow[] }> {
  const form = new FormData();
  images.forEach((b, i) => form.append("images", b, `ref_${i}.png`));
  try {
    const res = await fetch(`${API_BASE}/campaigns/${id}/badges`, { method: "POST", body: form });
    return await parse<Campaign & { uploads: UploadRow[] }>(res);
  } catch (err) {
    return failedToReach(err);
  }
}

export async function deleteBadgeRef(campaignId: number, badgeId: number): Promise<void> {
  try {
    await parse(
      await fetch(`${API_BASE}/campaigns/${campaignId}/badges/${badgeId}`, { method: "DELETE" }),
    );
  } catch (err) {
    return failedToReach(err);
  }
}

export const assetUrl = (path: string) => `${API_BASE}${path}`;
