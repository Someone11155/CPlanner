//
//  DesignTokens.swift
//  CPlanner
//
//  Todomate-inspired design tokens (vault://wiki/design/DESIGN_todomate.md).
//  Purely dark-mode design — app forces .preferredColorScheme(.dark) at root.
//
//  Core DNA adopted from Todomate:
//    - Pure-black canvas, no light-mode counterpart
//    - Squircle (continuous superellipse) cells for date grid + checkboxes
//    - Bright yellow (#FACC15) as universal completion / has-events accent
//    - Capsule pills for "named container" elements (buttons, group chips)
//    - White-pill / circle inversion for active / today states
//    - Saturated low-luminance accent text on group labels (TEXT only,
//      except for small ≤40px icon-circle backgrounds)
//
//  Skipped from Todomate (not applicable to a single-user productivity app):
//    - Friend avatar capsule row
//    - Activity feed pattern (avatar + heart)
//    - AI gradient logo
//    - Bottom action sheet for per-task actions (CPlanner uses an inline
//      folder menu instead — could be added later if desired)
//

import SwiftUI

// MARK: - Colors

extension Color {
    // Backgrounds — pure-black canvas + tiered dark grays
    /// True near-black background — the signature Todomate field.
    static let tdmCanvas        = Color(red: 0/255,   green: 0/255,   blue: 0/255)
    /// One shade lighter than canvas — squircle date cells, inactive pill bg.
    static let tdmCapsule       = Color(red: 28/255,  green: 28/255,  blue: 30/255)   // #1C1C1E
    /// Slightly more lifted — group label pills, segmented inactive tabs.
    static let tdmCapsuleSoft   = Color(red: 42/255,  green: 42/255,  blue: 42/255)   // #2A2A2A
    /// Pure white — active friend pill / today circle / primary CTA pill.
    static let tdmCapsuleActive = Color.white
    /// Floating menu / bottom-sheet container background.
    static let tdmBgMenu        = Color(red: 31/255,  green: 31/255,  blue: 34/255)   // #1F1F22

    // Yellow — the universal completion / emphasis accent
    /// Saturated yellow used for completed task checkboxes AND has-events date cells.
    static let tdmYellow        = Color(red: 250/255, green: 204/255, blue: 21/255)   // #FACC15

    // Date weekday colors — Korean calendar convention
    static let tdmDateSaturday  = Color(red: 10/255,  green: 132/255, blue: 255/255)  // #0A84FF
    static let tdmDateSunday    = Color(red: 255/255, green: 59/255,  blue: 48/255)   // #FF3B30

    // Group accent colors (used as TEXT only, on `tdmCapsuleSoft` pills)
    static let tdmAccentFriend  = Color(red: 63/255,  green: 174/255, blue: 232/255)  // teal-blue
    static let tdmAccentStudy   = Color(red: 92/255,  green: 209/255, blue: 124/255)  // mint-green
    static let tdmAccentSchool  = Color(red: 167/255, green: 139/255, blue: 250/255)  // lavender
    static let tdmAccentMisc    = Color(red: 156/255, green: 163/255, blue: 175/255)  // neutral

    // Icon circle accents (saturated FILL ≤ 40px circles — exception to "TEXT only" rule)
    static let tdmIconMemo            = Color(red: 250/255, green: 204/255, blue: 21/255)   // yellow
    static let tdmIconAlarm           = Color(red: 236/255, green: 72/255,  blue: 153/255)  // pink/magenta
    static let tdmIconTimer           = Color(red: 34/255,  green: 211/255, blue: 238/255)  // cyan
    static let tdmIconPhoto           = Color(red: 34/255,  green: 197/255, blue: 94/255)   // green
    static let tdmIconRepeatTomorrow  = Color(red: 249/255, green: 115/255, blue: 22/255)   // orange
    static let tdmIconRepeatOther     = Color(red: 239/255, green: 68/255,  blue: 68/255)   // red
    static let tdmIconDateChange      = Color(red: 59/255,  green: 130/255, blue: 246/255)  // blue
    static let tdmIconRoutine         = Color(red: 20/255,  green: 184/255, blue: 166/255)  // teal

    // Text — bright white primary, tiered grays for hierarchy
    static let tdmInkPrimary    = Color.white
    static let tdmInkBio        = Color(red: 161/255, green: 161/255, blue: 170/255)  // #A1A1AA
    static let tdmInkSecondary  = Color(red: 113/255, green: 113/255, blue: 122/255)  // #71717A
    static let tdmInkTertiary   = Color(red: 82/255,  green: 82/255,  blue: 91/255)   // #52525B
    static let tdmInkOnLight    = Color.black

    // Semantic
    static let tdmNotificationDot = Color(red: 239/255, green: 68/255, blue: 68/255)  // #EF4444
}

// MARK: - Spacing

enum DesignSpacing {
    static let xxs: CGFloat = 4
    static let xs:  CGFloat = 8
    static let sm:  CGFloat = 12
    static let md:  CGFloat = 16
    static let lg:  CGFloat = 24
    static let xl:  CGFloat = 32
    static let xxl: CGFloat = 48
    static let section: CGFloat = 64
}

// MARK: - Border Radius

enum DesignRadius {
    static let xs: CGFloat = 6
    static let sm: CGFloat = 8
    static let md: CGFloat = 12
    static let lg: CGFloat = 16
    static let xl: CGFloat = 20
    static let actionTile: CGFloat = 18
    /// Use this with `RoundedRectangle(cornerRadius:, style: .continuous)` to get
    /// a Todomate-style squircle. Standard `cornerRadius` (without `.continuous`)
    /// produces circular corners which look subtly wrong vs Todomate's character.
    static let squircle: CGFloat = 10
    static let full: CGFloat = 9999
}

// MARK: - Typography

/// Pretendard / Apple SD Gothic Neo / SF Pro fallback chain. The system font
/// (SF Pro) on macOS is humanist-geometric in the same family as Pretendard.
enum DesignFont {
    static func heading1(_ size: CGFloat = 28) -> Font { .system(size: size, weight: .bold,     design: .default) }
    static func heading2(_ size: CGFloat = 22) -> Font { .system(size: size, weight: .bold,     design: .default) }
    static func heading3(_ size: CGFloat = 18) -> Font { .system(size: size, weight: .semibold, design: .default) }
    static func body(_ size: CGFloat = 16)        -> Font { .system(size: size, weight: .regular,  design: .default) }
    static func bodyMedium(_ size: CGFloat = 16)  -> Font { .system(size: size, weight: .medium,   design: .default) }
    static func bodySmall(_ size: CGFloat = 14)   -> Font { .system(size: size, weight: .regular,  design: .default) }
    static func bodySmallMedium(_ size: CGFloat = 14) -> Font { .system(size: size, weight: .medium, design: .default) }
    static func caption(_ size: CGFloat = 13)     -> Font { .system(size: size, weight: .semibold, design: .default) }
    /// Tabular figures for date grid alignment.
    static func calendarDate(_ size: CGFloat = 14) -> Font {
        .system(size: size, weight: .medium, design: .default).monospacedDigit()
    }
    static func calendarMonth(_ size: CGFloat = 22) -> Font { .system(size: size, weight: .bold, design: .default) }
    static func button(_ size: CGFloat = 15)      -> Font { .system(size: size, weight: .semibold, design: .default) }
}

// MARK: - View Modifiers / Helpers

/// Todomate `friend-capsule-active` / `pill-tab-active` — white pill with black text.
/// Use for primary CTAs (the "active inversion" pattern).
struct PrimaryPillModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(DesignFont.button())
            .foregroundColor(.tdmInkOnLight)
            .padding(.vertical, 10)
            .padding(.horizontal, 18)
            .background(Color.tdmCapsuleActive)
            .clipShape(Capsule())
    }
}

/// Todomate `group-pill` style — soft dark gray pill, white-or-accent label text.
struct SecondaryPillModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(DesignFont.button())
            .foregroundColor(.tdmInkPrimary)
            .padding(.vertical, 10)
            .padding(.horizontal, 18)
            .background(Color.tdmCapsuleSoft)
            .clipShape(Capsule())
    }
}

/// Standard squircle cell (continuous superellipse). Use as background shape,
/// not just `cornerRadius` — the continuous style is a critical part of the brand feel.
struct SquircleBackgroundModifier: ViewModifier {
    var fill: Color
    var radius: CGFloat = DesignRadius.squircle
    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: radius, style: .continuous).fill(fill)
            )
    }
}

extension View {
    func primaryPill() -> some View { modifier(PrimaryPillModifier()) }
    func secondaryPill() -> some View { modifier(SecondaryPillModifier()) }
}
