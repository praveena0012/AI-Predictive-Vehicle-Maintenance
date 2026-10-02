const AUTH_API_URL = "http://127.0.0.1:8000";
const originalFetch = window.fetch.bind(window);

window.fetch = function(resource, options = {}) {

    const token =
        localStorage.getItem("buspredict_token");

    const url =
        typeof resource === "string"
            ? resource
            : resource.url;

    if (
        token &&
        url.startsWith(AUTH_API_URL)
    ) {

        const headers =
            new Headers(options.headers || {});

        if (!headers.has("Authorization")) {

            headers.set(
                "Authorization",
                `Bearer ${token}`
            );
        }

        options = {
            ...options,
            headers: headers
        };
    }

    return originalFetch(
        resource,
        options
    );
};
async function protectPage() {

    const token =
        localStorage.getItem("buspredict_token");

    if (!token) {
        window.location.href = "login.html";
        return;
    }

    try {

        const response = await fetch(
            `${AUTH_API_URL}/auth/check`,
            {
                headers: {
                    "Authorization":
                        `Bearer ${token}`
                }
            }
        );

        if (!response.ok) {

            localStorage.removeItem(
                "buspredict_token"
            );

            localStorage.removeItem(
                "buspredict_username"
            );

            window.location.href =
                "login.html";
        }

    } catch (error) {

        console.error(
            "Authentication check failed:",
            error
        );

        window.location.href =
            "login.html";
    }
}

protectPage();
async function logoutUser() {

    const token =
        localStorage.getItem("buspredict_token");

    try {

        if (token) {

            await fetch(
                `${AUTH_API_URL}/logout`,
                {
                    method: "POST",
                    headers: {
                        "Authorization":
                            `Bearer ${token}`
                    }
                }
            );

        }

    } catch (error) {

        console.error(
            "Logout failed:",
            error
        );

    } finally {

        localStorage.removeItem(
            "buspredict_token"
        );

        localStorage.removeItem(
            "buspredict_username"
        );

        window.location.href =
            "login.html";
    }
}