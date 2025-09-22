import i18n from "i18next";
import { initReactI18next } from "react-i18next";

const resources = {
  en: {
    translation: {
      disasterPrediction: "Disaster Prediction",
      dashboard: "Dashboard",
      floatchat: "FloatChat",
      profile: "Profile",
      history: "History",
      chat: "Chat",
      logout: "Logout",
    },
  },
  hi: {
    translation: {
      disasterPrediction: "आपदा भविष्यवाणी",
      dashboard: "डैशबोर्ड",
      floatchat: "फ्लोटचैट",
      profile: "प्रोफ़ाइल",
      history: "इतिहास",
      chat: "चैट",
      logout: "लॉगआउट",
    },
  },
};

i18n.use(initReactI18next).init({
  resources,
  lng: "en", // default language
  fallbackLng: "en",
  interpolation: {
    escapeValue: false,
  },
});

export default i18n;
