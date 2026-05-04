//
//  DesignTokens.swift
//  CPlanner
//
//  Notion-inspired design tokens. Reference: vault://wiki/design/DESIGN_notion.md
//  Selective application — Notion's marketing-page DNA mapped to a productivity app surface.
//  Skipped from Notion: hero navy band, pastel feature-card tints, decorative dots/wires.
//  Adopted: color palette (purple primary, warm charcoal text, hairline borders, surface),
//  8px button radius / 12px card radius, 4px base spacing, subtle Level-1/Level-2 shadows.
//

import SwiftUI

// MARK: - Colors

extension Color {
    /// Notion signature purple — primary CTA. Reserved for the dominant action.
    /// Notion's actual brand purple ≈ rgb(80, 70, 228).
    static let notionPurple = Color(red: 80/255, green: 70/255, blue: 228/255)
    static let notionPurplePressed = Color(red: 64/255, green: 54/255, blue: 200/255)

    /// Warm charcoal — Notion's signature body text color (not pure black).
    static let notionInk = Color(red: 55/255, green: 53/255, blue: 47/255)        // #37352F
    static let notionInkDeep = Color(red: 25/255, green: 25/255, blue: 25/255)    // near-black emphasis

    /// Secondary / tertiary text grays.
    static let notionSlate = Color(red: 120/255, green: 119/255, blue: 116/255)   // body secondary
    static let notionSteel = Color(red: 155/255, green: 154/255, blue: 151/255)   // tertiary, footer
    static let notionStone = Color(red: 174/255, green: 173/255, blue: 169/255)   // muted labels
    static let notionMuted = Color(red: 199/255, green: 198/255, blue: 195/255)   // disabled

    /// Surfaces — page bg, soft section bg.
    /// Use `notionSurface` instead of pure white for soft section divisions.
    static let notionCanvas = Color(red: 255/255, green: 255/255, blue: 255/255)  // primary card
    static let notionSurface = Color(red: 247/255, green: 246/255, blue: 243/255) // section bg, search rest
    static let notionSurfaceSoft = Color(red: 252/255, green: 251/255, blue: 250/255)

    /// Borders / dividers.
    static let notionHairline = Color(red: 233/255, green: 232/255, blue: 229/255)
    static let notionHairlineStrong = Color(red: 219/255, green: 218/255, blue: 215/255)

    // Dark-mode variants — Notion's dark theme. macOS dark mode automatically picks these via dynamic colors below.
}

/// Adaptive (light/dark) versions of the most-used tokens. Use these in views to get
/// automatic dark-mode response. The plain Color extensions above are the light-mode source.
extension Color {
    static var notionInkAdaptive: Color {
        Color(NSColor(name: nil, dynamicProvider: { appearance in
            appearance.bestMatch(from: [.darkAqua, .vibrantDark]) != nil
                ? NSColor(white: 0.92, alpha: 1.0)
                : NSColor(red: 55/255, green: 53/255, blue: 47/255, alpha: 1.0)
        }))
    }
    static var notionSlateAdaptive: Color {
        Color(NSColor(name: nil, dynamicProvider: { appearance in
            appearance.bestMatch(from: [.darkAqua, .vibrantDark]) != nil
                ? NSColor(white: 0.62, alpha: 1.0)
                : NSColor(red: 120/255, green: 119/255, blue: 116/255, alpha: 1.0)
        }))
    }
    static var notionCanvasAdaptive: Color {
        Color(NSColor(name: nil, dynamicProvider: { appearance in
            appearance.bestMatch(from: [.darkAqua, .vibrantDark]) != nil
                ? NSColor(white: 0.10, alpha: 1.0)
                : NSColor.white
        }))
    }
    static var notionSurfaceAdaptive: Color {
        Color(NSColor(name: nil, dynamicProvider: { appearance in
            appearance.bestMatch(from: [.darkAqua, .vibrantDark]) != nil
                ? NSColor(white: 0.14, alpha: 1.0)
                : NSColor(red: 247/255, green: 246/255, blue: 243/255, alpha: 1.0)
        }))
    }
    static var notionHairlineAdaptive: Color {
        Color(NSColor(name: nil, dynamicProvider: { appearance in
            appearance.bestMatch(from: [.darkAqua, .vibrantDark]) != nil
                ? NSColor(white: 1.0, alpha: 0.10)
                : NSColor(red: 233/255, green: 232/255, blue: 229/255, alpha: 1.0)
        }))
    }
}

// MARK: - Spacing

/// 4px base unit, 8px primary increment. Use these values directly via padding modifiers.
enum DesignSpacing {
    static let xxs: CGFloat = 4
    static let xs: CGFloat = 8
    static let sm: CGFloat = 12
    static let md: CGFloat = 16
    static let lg: CGFloat = 24
    static let xl: CGFloat = 32
    static let xxl: CGFloat = 48
    static let section: CGFloat = 64
}

// MARK: - Border Radius

enum DesignRadius {
    static let xs: CGFloat = 4    // Tag chips
    static let sm: CGFloat = 6    // Type badges
    static let md: CGFloat = 8    // Buttons, inputs
    static let lg: CGFloat = 12   // Cards, panels
    static let xl: CGFloat = 16   // Larger feature panels
}

// MARK: - Typography

/// Notion-Sans is Inter-based. macOS system font (SF Pro) shares the humanist-geometric
/// character closely enough — using `.system` keeps native appearance while matching
/// Notion's typographic scale. If Inter is bundled later, swap `.system` → `Font.custom("Inter-...", size:)`.
enum DesignFont {
    static func heading1(_ size: CGFloat = 28) -> Font { .system(size: size, weight: .semibold, design: .default) }
    static func heading2(_ size: CGFloat = 22) -> Font { .system(size: size, weight: .semibold, design: .default) }
    static func heading3(_ size: CGFloat = 18) -> Font { .system(size: size, weight: .semibold, design: .default) }
    static func bodyMedium(_ size: CGFloat = 16) -> Font { .system(size: size, weight: .medium, design: .default) }
    static func body(_ size: CGFloat = 16) -> Font { .system(size: size, weight: .regular, design: .default) }
    static func bodySmall(_ size: CGFloat = 14) -> Font { .system(size: size, weight: .regular, design: .default) }
    static func bodySmallMedium(_ size: CGFloat = 14) -> Font { .system(size: size, weight: .medium, design: .default) }
    static func caption(_ size: CGFloat = 13) -> Font { .system(size: size, weight: .semibold, design: .default) }
    static func button(_ size: CGFloat = 14) -> Font { .system(size: size, weight: .medium, design: .default) }
}

// MARK: - View Modifiers

/// Notion `card-base` — white surface, 12px rounded, hairline border, 24px padding.
struct CardBaseModifier: ViewModifier {
    var padding: CGFloat = DesignSpacing.lg
    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(Color.notionCanvasAdaptive)
            .cornerRadius(DesignRadius.lg)
            .overlay(
                RoundedRectangle(cornerRadius: DesignRadius.lg)
                    .stroke(Color.notionHairlineAdaptive, lineWidth: 1)
            )
    }
}

/// Notion `card-base` with subtle Level-1 shadow.
struct CardElevatedModifier: ViewModifier {
    var padding: CGFloat = DesignSpacing.lg
    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(Color.notionCanvasAdaptive)
            .cornerRadius(DesignRadius.lg)
            .overlay(
                RoundedRectangle(cornerRadius: DesignRadius.lg)
                    .stroke(Color.notionHairlineAdaptive, lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.04), radius: 2, x: 0, y: 1)
    }
}

/// Notion `button-primary` — purple rectangular CTA, 8px rounded, 10px 18px padding.
struct PrimaryButtonModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(DesignFont.button())
            .foregroundColor(.white)
            .padding(.vertical, 10)
            .padding(.horizontal, 18)
            .background(Color.notionPurple)
            .cornerRadius(DesignRadius.md)
    }
}

/// Notion `button-secondary` — outlined transparent.
struct SecondaryButtonModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(DesignFont.button())
            .foregroundColor(.notionInkAdaptive)
            .padding(.vertical, 10)
            .padding(.horizontal, 18)
            .background(Color.clear)
            .overlay(
                RoundedRectangle(cornerRadius: DesignRadius.md)
                    .stroke(Color.notionHairlineStrong, lineWidth: 1)
            )
    }
}

extension View {
    func cardBase(padding: CGFloat = DesignSpacing.lg) -> some View {
        modifier(CardBaseModifier(padding: padding))
    }
    func cardElevated(padding: CGFloat = DesignSpacing.lg) -> some View {
        modifier(CardElevatedModifier(padding: padding))
    }
    func primaryButtonStyle() -> some View {
        modifier(PrimaryButtonModifier())
    }
    func secondaryButtonStyle() -> some View {
        modifier(SecondaryButtonModifier())
    }
}
